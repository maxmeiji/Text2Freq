import torch
import torch.nn as nn
import torch.nn.functional as F

class RevIN(nn.Module):
    def __init__(self, eps=1e-5):
        super(RevIN, self).__init__()
        self.num_features = 1
        self.eps = eps


    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            x = self._denormalize_mean(x)
        return x
    
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean
        return x
    def _denormalize_mean(self, x):
        x = x + self.mean
        return x

class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, hidden_dim):
        """
        Args:
            query_dim: Dimension of the query (text_pred) features
            key_dim: Dimension of the key (series_pred) features
            value_dim: Dimension of the value (series_pred) features
            hidden_dim: Dimension of the attention output
        """
        super(CrossAttention, self).__init__()
        # Linear layers for query, key, value projections
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(value_dim, hidden_dim)
        
        # Output projection after attention
        self.output_proj = nn.Linear(hidden_dim, value_dim)

    def forward(self, text_pred, series_pred):
        """
        Args:
            text_pred: Tensor of shape [batch_size, text_len] (query)
            series_pred: Tensor of shape [batch_size, series_len, series_dim] (key, value)
        
        Returns:
            output: Tensor of shape [batch_size, series_len, value_dim]
        """
        # Project text (query) and series (key, value) to hidden_dim
        # Project inputs into hidden space
        query = self.query_proj(text_pred).unsqueeze(1)  # [batch_size, 1, hidden_dim]
        key = self.key_proj(series_pred).unsqueeze(2)    # [batch_size, hidden_dim, 1]
        value = self.value_proj(series_pred)             # [batch_size, value_dim]

        # Compute attention scores (dot product of query and key)
        attn_scores = torch.bmm(query, key).squeeze(2)  # [batch_size, 1, 1] → [batch_size, 1]
        attn_weights = F.softmax(attn_scores, dim=-1)   # [batch_size, 1]

        # Apply attention weights to value
        output = attn_weights * value  # Element-wise multiplication [batch_size, value_dim]
        return output

class Attention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention, self).__init__()
        self.query = nn.Linear(input_dim, hidden_dim)
        self.key = nn.Linear(input_dim, hidden_dim)
        self.value = nn.Linear(input_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_weights = F.softmax(Q @ K.transpose(-2, -1) / self.scale, dim=-1)
        return attention_weights @ V
           
class Attention1(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Attention1, self).__init__()
        self.attn_fc = nn.Linear(input_dim, hidden_dim)  # Learnable attention weights
        self.attn_score = nn.Linear(hidden_dim, 1)  # Scalar attention score
        self.projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, fused_features):
        """
        Args:
            fused_features: Tensor of shape [batch_size, 2 * hidden_dim]
        
        Returns:
            attention_out: Tensor of shape [batch_size, hidden_dim]
        """
        attn_weights = torch.tanh(self.attn_fc(fused_features))  # Non-linear transformation
        attn_scores = self.attn_score(attn_weights)  # Compute scalar scores
        attn_weights = F.softmax(attn_scores, dim=1)  # Normalize scores

        # Weighted sum of input features
        attention_out = fused_features * attn_weights  # Element-wise scaling
        attention_out = self.projection(attention_out)  # Now (batch, hidden_dim)
        return attention_out
                
class TTS(nn.Module):
    def __init__(self, series_length, pred_len):
        super(TTS, self).__init__()
        
        self.series_length = series_length
        self.pred_len = pred_len
        self.hidden = pred_len*2
        # the serues lenght is 6  here
        # Define your transformers and tokenizers here
        self.revin = RevIN()
        self.revin2 = RevIN()

        # Feature extraction for series_pred 
        self.pred1_fc1 = nn.Linear(pred_len, self.hidden)
        self.pred1_fc2 = nn.Linear(self.hidden, self.hidden)
        self.pred1_dropout = nn.Dropout(p=0.3) 
        self.pred1_bn1 = nn.BatchNorm1d(self.hidden)
        self.pred1_bn2 = nn.BatchNorm1d(self.hidden)

        # Feature extraction for text_pred
        self.pred2_fc1 = nn.Linear(self.series_length, self.hidden)
        self.pred2_fc2 = nn.Linear(self.hidden, self.hidden)
        self.pred2_dropout = nn.Dropout(p=0.3)
        self.pred2_bn1 = nn.BatchNorm1d(self.hidden)
        self.pred2_bn2 = nn.BatchNorm1d(self.hidden)

        # Attention
        self.attention = Attention(2 * self.hidden, self.hidden)
        self.fusion_fc1 = nn.Linear(self.hidden, self.hidden // 2)
        self.fusion_dropout = nn.Dropout(p=0.3)
        self.fusion_fc2 = nn.Linear(self.hidden // 2, pred_len)
        self.fusion_bn = nn.BatchNorm1d(pred_len)

    def forward(self, text_pred, series_pred):
        series_pred = self.revin(series_pred, mode='norm')[:,:, -1] 
        text_pred = self.revin2(text_pred, mode='norm')[:,:, -1] 
        # Feature extraction for series_pred
        pred1_out = torch.relu(self.pred1_fc1(series_pred))
        #pred1_out = self.pred1_bn1(pred1_out)  # BatchNorm
        pred1_out = self.pred1_dropout(pred1_out) 
        series_pred = torch.relu(self.pred1_fc2(pred1_out))
        #series_pred = self.pred1_bn2(series_pred)
        series_pred = self.pred1_dropout(series_pred) 
        
        # Feature extraction for text_pred
        pred2_out = torch.relu(self.pred2_fc1(text_pred))
        #pred2_out = self.pred2_bn1(pred2_out)  # BatchNorm
        pred2_out = self.pred2_dropout(pred2_out)  # Dropout
        text_pred = torch.relu(self.pred2_fc2(pred2_out))
        #text_pred = self.pred2_bn2(text_pred)  # BatchNorm
        text_pred = self.pred2_dropout(text_pred)


        # Concatenate the features
        fused_features = torch.cat((series_pred, text_pred), dim=1)

        # shape of fused_featured: [batch size, hidden shape*2]

        # Apply attention in the fusion layer
        attention_out = self.attention(fused_features)
        outputs = torch.relu(self.fusion_fc1(attention_out))
        #outputs = self.fusion_bn(outputs)  # BatchNorm
        outputs = self.fusion_dropout(outputs) 
        outputs = self.fusion_fc2(outputs)

        outputs = outputs.unsqueeze(2)
        outputs = self.revin(outputs, mode='denorm')
        return outputs, 0


class TTS_with_prior(nn.Module):
    def __init__(self, series_length, pred_len):
        super(TTS_with_prior, self).__init__()
        
        self.series_length = series_length
        self.pred_len = pred_len
        self.hidden = pred_len*2

        # Define your transformers and tokenizers here
        self.revin = RevIN()

        # Feature extraction for series_pred
        self.pred1_fc1 = nn.Linear(pred_len, pred_len*2)
        self.pred1_fc2 = nn.Linear(pred_len*2, self.hidden)

        # Feature extraction for text_pred
        self.pred2_fc1 = nn.Linear(self.series_length, pred_len)
        self.pred2_fc2 = nn.Linear(pred_len, self.hidden)

        # feature extraction for prior_y
        self.pred3_fc1 = nn.Linear(pred_len, pred_len*2)
        self.pred3_fc2 = nn.Linear(pred_len*2, self.hidden)

        # Attention - Final fusion
        self.attention1 = Attention(2 * self.hidden, self.hidden)
        self.fusion1_fc1 = nn.Linear(self.hidden, self.hidden)

        # Attention - Final fusion
        self.attention2 = Attention(2 * self.hidden, self.hidden)
        self.fusion2_fc1 = nn.Linear(self.hidden, self.hidden // 2)
        self.fusion2_fc2 = nn.Linear(self.hidden // 2, pred_len)

    def forward(self, text_pred, series_pred, prior_y):

        # Debugging
        prior_y = prior_y.squeeze(2)
        series_pred = self.revin(series_pred, mode='norm')[:,:, -1] 
        text_pred = text_pred.squeeze(2)

        # Feature extraction for series_pred
        pred1_out = torch.relu(self.pred1_fc1(series_pred))
        series_pred = torch.relu(self.pred1_fc2(pred1_out))
        
        # Feature extraction for text_pred
        pred2_out = torch.relu(self.pred2_fc1(text_pred))
        text_pred = torch.relu(self.pred2_fc2(pred2_out))

        # feature extraction for prior_y
        pred3_out = torch.relu(self.pred3_fc1(prior_y))
        prior_pred = torch.relu(self.pred3_fc2(pred3_out))

        # Concatenate the prior and text
        text_prior = torch.cat((prior_pred, text_pred), dim = 1)
        attention_out = self.attention1(text_prior)
        text_prior = torch.relu(self.fusion1_fc1(attention_out))


        # Concatenate the prior_text with series prediction
        fused_features = torch.cat((text_prior, series_pred), dim=1)
        attention_out = self.attention2(fused_features)
        fused_out_2 = torch.relu(self.fusion2_fc1(attention_out))
        output = self.fusion2_fc2(fused_out_2)
        output = output.unsqueeze(2)
        
        # Denormalize the output
        output = self.revin(output, mode='denorm')
        
        return output

class TTS_concat(nn.Module):
    def __init__(self, series_length, pred_len):
        super(TTS_concat, self).__init__()
        
        self.series_length = series_length
        self.pred_len = pred_len
        self.hidden = pred_len*2
        self.revin = RevIN()
        self.revin2 = RevIN()
        # Feature extraction for text_pred
        self.pred2_fc1 = nn.Linear(self.series_length, pred_len)
        self.pred2_fc2 = nn.Linear(pred_len, pred_len)

        self.pred1_dropout = nn.Dropout(p=0.3) 
        self.pred2_dropout = nn.Dropout(p=0.3) 
        # Concatenation - Final fusion
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, text_pred, series_pred):
        
        series_pred = self.revin(series_pred, mode='norm')
        # Feature extraction for text_pred
        text_pred = self.revin2(text_pred, mode='norm')
        text_pred = text_pred.squeeze(2)
        pred2_out = torch.relu(self.pred2_fc1(text_pred))
        pred2_out = self.pred1_dropout(pred2_out) 
        text_pred = torch.relu(self.pred2_fc2(pred2_out))
        text_pred = self.pred2_dropout(text_pred) 
        text_pred = text_pred.unsqueeze(2)
        

        # Concatenate the prior and text
        alpha = torch.sigmoid(self.alpha)
        outputs=(1-alpha)*series_pred+alpha*text_pred
        
        # Denormalize the output
        outputs = self.revin(outputs, mode='denorm')
        return outputs, alpha


class TTS_TimeMMD(nn.Module):
    def __init__(self, series_length, pred_len):
        super(TTS_TimeMMD, self).__init__()

        # Concatenation - Final fusion
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, text_pred, series_pred):

        # Concatenate the prior and text
        alpha = torch.sigmoid(self.alpha)
        outputs=(1-alpha)*series_pred+alpha*text_pred
        
        return outputs

class TTS_cross(nn.Module):
    def __init__(self, series_length, pred_len):
        super(TTS_cross, self).__init__()
        
        self.series_length = series_length
        self.pred_len = pred_len
        self.hidden = pred_len*2
        # the serues lenght is 6  here
        # Define your transformers and tokenizers here
        self.revin = RevIN()
        self.revin2 = RevIN()

        # Feature extraction for series_pred 
        self.pred1_fc1 = nn.Linear(pred_len, self.hidden)
        self.pred1_fc2 = nn.Linear(self.hidden, self.hidden)
        self.pred1_dropout = nn.Dropout(p=0.2) 
        self.pred1_bn1 = nn.BatchNorm1d(self.hidden)
        self.pred1_bn2 = nn.BatchNorm1d(self.hidden)

        # Feature extraction for text_pred
        self.pred2_fc1 = nn.Linear(self.series_length, self.hidden)
        self.pred2_fc2 = nn.Linear(self.hidden, self.hidden)
        self.pred2_dropout = nn.Dropout(p=0.3)
        self.pred2_bn1 = nn.BatchNorm1d(self.hidden)
        self.pred2_bn2 = nn.BatchNorm1d(self.hidden)

         # Layer Normalization for residual connections
        self.norm1 = nn.LayerNorm(self.hidden)
        self.norm2 = nn.LayerNorm(self.hidden)

        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden, self.hidden * 2),
            nn.ReLU(),
            nn.Linear(self.hidden * 2, self.hidden),
            nn.Dropout(0.1)  # Dropout for regularization
        )
        self.fusion_fc = nn.Linear(self.hidden, pred_len)
        # Attention
        # self.attention = Attention1(2 * self.hidden, self.hidden)
        self.cross_att1 = CrossAttention(self.series_length,self.pred_len,self.pred_len,self.hidden)
        self.cross_att2 = CrossAttention(self.hidden,self.pred_len,self.pred_len,self.hidden)
        self.fusion_fc1 = nn.Linear(self.hidden, self.hidden // 2)
        self.fusion_dropout = nn.Dropout(p=0.3)
        self.fusion_fc2 = nn.Linear(self.hidden // 2, pred_len)
        self.fusion_bn = nn.BatchNorm1d(pred_len)

    def forward(self, text_pred, series_pred):
        series_pred = self.revin(series_pred, mode='norm')
        series_pred = series_pred.squeeze(2)
        text_pred = text_pred.unsqueeze(2)
        text_pred = self.revin2(text_pred, mode='norm')
        text_pred = text_pred.squeeze(2)
        
        attn_out = self.cross_att1(text_pred, series_pred)
        text_proj = self.pred2_fc1(text_pred)
        # Add & Norm 1 (Residual Connection)
        attn_out1 = self.norm1(attn_out + text_proj)
        
        attn_out2 = self.cross_att2(attn_out1, series_pred) 
        attn_out2 = self.norm2(attn_out2 + attn_out1)  # Residual connection
        # Feed-Forward Network
        ffn_out = self.ffn(attn_out2)

        # Add & Norm 2 (Residual Connection)
        fusion_out = self.norm2(ffn_out + attn_out2)

        # Final Prediction
        prediction = self.fusion_fc(fusion_out)  # [batch_size, pred_length]

        outputs = prediction.unsqueeze(2)
        outputs = self.revin(outputs, mode='denorm')
        return outputs, 0

class TTS_cross_GT(nn.Module):
    def __init__(self, series_length, pred_len):
        super(TTS_cross_GT, self).__init__()
        
        self.series_length = series_length
        self.pred_len = pred_len
        self.hidden = pred_len*2
        # the serues lenght is 6  here
        # Define your transformers and tokenizers here
        self.revin = RevIN()
        self.revin2 = RevIN()
        self.revin3 = RevIN()

        # Feature extraction for series_pred 
        self.pred1_fc1 = nn.Linear(pred_len, self.hidden)
        self.pred1_fc2 = nn.Linear(self.hidden, self.hidden)
        self.pred1_dropout = nn.Dropout(p=0.2) 
        self.pred1_bn1 = nn.BatchNorm1d(self.hidden)
        self.pred1_bn2 = nn.BatchNorm1d(self.hidden)

        # Feature extraction for text_pred
        self.pred2_fc1 = nn.Linear(pred_len, self.hidden)
        self.pred2_fc2 = nn.Linear(self.hidden, self.hidden)
        self.pred2_dropout = nn.Dropout(p=0.3)
        self.pred2_bn1 = nn.BatchNorm1d(self.hidden)
        self.pred2_bn2 = nn.BatchNorm1d(self.hidden)

         # Layer Normalization for residual connections
        self.norm1 = nn.LayerNorm(self.hidden)
        self.norm2 = nn.LayerNorm(self.hidden)

        # Feed-Forward Network (FFN)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden, self.hidden * 2),
            nn.ReLU(),
            nn.Linear(self.hidden * 2, self.hidden),
            nn.Dropout(0.1)  # Dropout for regularization
        )
        self.fusion_fc = nn.Linear(self.hidden, pred_len)
        # Attention
        # self.attention = Attention1(2 * self.hidden, self.hidden)
        self.cross_att1 = CrossAttention(self.series_length,self.pred_len,self.pred_len,self.hidden)
        self.cross_att2 = CrossAttention(self.hidden,self.pred_len,self.pred_len,self.hidden)
        self.fusion_fc1 = nn.Linear(self.hidden, self.hidden // 2)
        self.fusion_dropout = nn.Dropout(p=0.3)
        self.fusion_fc2 = nn.Linear(self.hidden // 2, pred_len)
        self.fusion_bn = nn.BatchNorm1d(pred_len)

    def forward(self, text_pred, series_pred, ground_truth):

        series_pred = self.revin(series_pred, mode='norm')
        series_pred = series_pred.squeeze(2)
        #text_pred = text_pred.unsqueeze(2)
        #text_pred = self.revin2(text_pred, mode='norm')
        #text_pred = text_pred.squeeze(2)
        ground_truth = self.revin3(ground_truth, mode='norm')
        ground_truth = ground_truth.squeeze(2)

        attn_out = self.cross_att1(text_pred, series_pred)
        text_proj = self.pred2_fc1(text_pred)
        # Add & Norm 1 (Residual Connection)
        attn_out1 = self.norm1(attn_out + text_proj)
        
        attn_out2 = self.cross_att2(attn_out1, series_pred) 
        attn_out2 = self.norm2(attn_out2 + attn_out1)  # Residual connection
        # Feed-Forward Network
        ffn_out = self.ffn(attn_out2)

        # Add & Norm 2 (Residual Connection)
        fusion_out = self.norm2(ffn_out + attn_out2)

        # Final Prediction
        prediction = self.fusion_fc(fusion_out)  # [batch_size, pred_length]

        outputs = prediction.unsqueeze(2)
        outputs = self.revin(outputs, mode='denorm')

        # return params
        series_pred = series_pred.unsqueeze(2)
        series_pred = self.revin(series_pred, mode='denorm')
        series_pred = series_pred.squeeze(2)

        text_pred = text_pred.unsqueeze(2)
        #text_pred = self.revin3(text_pred, mode='denorm')
        text_pred = text_pred.squeeze(2)

        ground_truth = ground_truth.unsqueeze(2)
        ground_truth = self.revin3(ground_truth, mode='denorm')
        ground_truth = ground_truth.squeeze(2)

        return outputs, series_pred, text_pred, ground_truth