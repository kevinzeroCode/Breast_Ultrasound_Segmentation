from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model
from .layers import SelfAttention

def conv_block(input_tensor, num_filters):
    """標準卷積區塊: Conv -> ReLU -> Conv -> ReLU"""
    x = Conv2D(num_filters, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(input_tensor)
    x = Conv2D(num_filters, (3, 3), activation="relu", padding="same", kernel_initializer='he_normal')(x)
    return x

def encoder_block(input_tensor, num_filters):
    """編碼區塊: Conv Block -> MaxPool"""
    x = conv_block(input_tensor, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(input_tensor, skip_features, num_filters, use_attention=False):
    """解碼區塊: UpConv -> (Optional Attention) -> Concat -> Conv Block"""
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input_tensor)
    
    if use_attention:
        # 這是你論文的核心：在 Skip Connection 結合處加入 Attention
        attention_layer = SelfAttention(num_filters)
        x_att = attention_layer(x)
        # 注意：根據你的原始碼，這裡是先 concat skip 和 upconv，然後再 concat attention 結果
        # 但這部分邏輯有點特別，照你的原始碼還原：
        x = concatenate([x, skip_features])
        x = concatenate([x, x_att])
    else:
        x = concatenate([x, skip_features])
        
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape, use_attention=False):
    """
    建立模型
    use_attention=True  -> 回傳你的 Proposed Method (Attention U-Net)
    use_attention=False -> 回傳 Baseline (Standard U-Net)
    """
    inputs = Input(input_shape)
    
    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    # Bridge
    b1 = conv_block(p4, 1024)

    # Decoder
    d1 = decoder_block(b1, s4, 512, use_attention)
    d2 = decoder_block(d1, s3, 256, use_attention)
    d3 = decoder_block(d2, s2, 128, use_attention)
    d4 = decoder_block(d3, s1, 64, use_attention)
    
    # Output
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
    
    model_name = "Attention_UNet" if use_attention else "Standard_UNet"
    model = Model(inputs, outputs, name=model_name)
    return model