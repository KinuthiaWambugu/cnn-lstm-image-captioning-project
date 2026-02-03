"""
Image Captioning Script - loads pre-trained captioning model and generate captions.
Required: model.weights.h5, vocab.json

"""
import tensorflow as tf
import numpy as np
import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

#configuration
MAX_LENGTH = 40
VOCABULARY_SIZE = 15000
EMBEDDING_DIM = 512
UNITS = 512
NUM_HEADS_ENCODER = 1
NUM_HEADS_DECODER = 8

#Model architecture
#combine token embeddings and positional embeddings
class Embeddings(tf.keras.layers.Layer):
    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(
            input_dim=max_len, output_dim=embed_dim
        )
    
    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]
        position_ids = tf.range(start=0, limit=length, delta=1)
        position_ids = tf.expand_dims(position_ids, axis=0)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        return token_embeddings + position_embeddings
    
#refine images features using self-attention
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads, key_dim=embed_dim
        )
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")

    def call(self, x, training=False):
        x = self.layer_norm_1(x)
        x = self.dense(x)
        attn_output = self.attention(
            query = x,
            value = x,
            key = x,
            attention_mask = None,
            training = training
        )
        x = self.layer_norm_2(x + attn_output)
        return x

#generate captions word by word using self and cross-attention
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_dim, units, num_heads, vocab_size, max_length):
        super().__init__()
        self.embedding = Embeddings(vocab_size, embed_dim, max_length) #check this
        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads= num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads = num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()

        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)
        self.out = tf.keras.layers.Dense(vocab_size, activation="softmax")
        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)

    def call(self, input_ids, encoder_output, training, mask=None):
        embeddings = self.embedding(input_ids)
        combined_mask = None
        padding_mask = None

        if mask is not None:
            causal_mask = self.get_causal_attention_mask(embeddings)
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)
        
        #self attention
        attn_output_1 = self.attention_1(
            query = embeddings,
            value=embeddings,
            key=embeddings,
            attention_mask = combined_mask,
            training = training
        )
        out_1 = self.layernorm_1(embeddings + attn_output_1)

        #cross-attention
        attn_output_2 = self.attention_2(
            query = out_1,
            value = encoder_output,
            key = encoder_output,
            attention_mask = padding_mask,
            training = training
        )
        out_2 = self.layernorm_2(out_1 + attn_output_2)

        #feed forward
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)
        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)
        preds = self.out(ffn_out)
        return preds

    #causal mask to prevent attending to future tokens
    def get_causal_attention_mask(self, inputs):
        seq_len = tf.shape(inputs)[1]
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        return tf.expand_dims(mask, 0)

#InceptionV3 encoder
def CNN_Encoder():
    inception_v3 = tf.keras.applications.InceptionV3(
        include_top = False,
        weights = "imagenet",
        input_shape =(299,299,3)
    )

    inception_v3.trainable = False
    output = inception_v3.output
    output = tf.keras.layers.Reshape((-1, output.shape[-1]))(output)
    cnn_model = tf.keras.models.Model(inception_v3.input, output)
    return cnn_model

#complete image captioning: cnn model + transformer encoder + decoder
class ImageCaptioningModel(tf.keras.Model):
    def __init__(self, cnn_model, encoder, decoder):
        super().__init__()
        self.cnn_model= cnn_model
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        if isinstance(inputs, tuple):
            imgs, captions = inputs
        else:
            imgs = inputs
            captions = None
        
        img_embed = self.cnn_model(imgs)
        encoder_output = self.encoder(img_embed, training=False)
        if captions is not None:
            captions = tf.convert_to_tensor(captions)
            if len(captions.shape) == 1:
                captions = tf.expand_dims(captions, 0)
            y_input = captions[:, :-1]
            mask = (y_input != 0)
            y_pred = self.decoder(
                y_input, encoder_output, training=False, mask=mask
            )
            return y_pred

        return encoder_output

#load vocabulary
def load_vocab(vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    word2idx = tf.keras.layers.StringLookup(
        vocabulary=vocab,
        mask_token = ""
    )

    idx2word = tf.keras.layers.StringLookup(
        vocabulary = vocab,
        mask_token="",
        invert=True
    )
    print(f"Loaded vocabulary with {len(vocab)} tokens")
    print(vocab[:10])
    print("Last 10 tokens:", vocab[-10:])
    return word2idx, idx2word, vocab

#rebuild tokenizer with saved vocab
"""
def build_tokenizer(vocab):
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens = None,
        standardize = None,
        output_sequence_length = MAX_LENGTH
    )
    tokenizer.set_vocabulary(vocab)
    return tokenizer
"""

#load and preprocess image
def load_and_preprocess_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels = 3)
    img = tf.keras.layers.Resizing(299,299)(img)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img

#generate caption for image
def generate_caption(image_path, caption_model, word2idx, idx2word, max_length=MAX_LENGTH, temperature=1.0, add_noise=False):
    img = load_and_preprocess_image(image_path)
    img = tf.expand_dims(img, axis=0)
    img_embed = caption_model.cnn_model(img)
    img_encoded = caption_model.encoder(img_embed, training=False)
    caption_tokens = ["[start]"]

    for i in range(max_length - 1):
        token_ids = word2idx(tf.constant([" ".join(caption_tokens)]))
        if len(token_ids.shape) == 1:
            token_ids = tf.expand_dims(token_ids, 0)
        token_ids = token_ids[:, :-1]
        mask = tf.cast(token_ids != 0, tf.int32)

        preds = caption_model.decoder(
            token_ids,
            img_encoded,
            training=False,
            mask=mask
        )
        logits = preds[0, i, :]
        if temperature != 1.0:
            logits = logits / temperature
        pred_id = tf.argmax(logits).numpy()
        pred_word = idx2word(pred_id).numpy().decode("utf-8")
        if pred_word == "[end]":
            break
        caption_tokens.append(pred_word)

    return " ".join(caption_tokens[1:])

#display captions and image
def display_image_with_cap(image_path, caption):
    img = plt.imread(image_path)
    plt.figure(figsize=(10,6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Caption: {caption}", fontsize=12, wrap=True)
    plt.tight_layout()
    plt.show()


#load model
def load_model(weights_path, vocab_path):
    """
    Load the complete model with saved weights and vocabulary.
    """
    print("Loading vocabulary...")
    word2idx, idx2word, vocab = load_vocab(vocab_path)
    vocab_size = len(vocab)
    bad_tokens = [t for t in vocab if t.isdigit()]
    print("Numeric tokens:", bad_tokens[:20])

    print("Building model architecture...")
    # Build encoder
    encoder = TransformerEncoderLayer(EMBEDDING_DIM, NUM_HEADS_ENCODER)
    
    # Build decoder with correct vocab size from tokenizer
    decoder = TransformerDecoderLayer(
        EMBEDDING_DIM, 
        UNITS, 
        NUM_HEADS_DECODER,
        vocab_size=vocab_size,
        max_length=MAX_LENGTH
    )
    
    # Build CNN encoder
    cnn_model = CNN_Encoder()
    
    # Build complete model
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model,
        encoder=encoder,
        decoder=decoder
    )
    
    # Build the model by doing a forward pass with dummy data
    print("Initializing model with dummy data...")
    dummy_img = tf.zeros((1, 299, 299, 3))
    dummy_caption = tf.zeros((1, MAX_LENGTH), dtype=tf.int32)
    _ = caption_model((dummy_img, dummy_caption))
    
    print(f"Loading weights from {weights_path}...")
    caption_model.load_weights(
        weights_path,
        skip_mismatch = True
    )
    
    print("Model loaded successfully!")
    print(f"Vocabulary size: {vocab_size}")
    
    return caption_model, word2idx, idx2word

def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for images"
    )
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to image file'
    )
    parser.add_argument(
        '--weights',
        type=str,
        required=True,
        help='Path to model weights file'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        required=True,
        help='Path to vocab JSON file'
    )
    parser.add_argument(
        '--temperature',
        type=int,
        default=1.0,
        help='Sampling temp'
    )
    parser.add_argument(
        '--display',
        action='store_true',
        help='Display image with caption'
    )

    args = parser.parse_args()
    caption_model, word2idx, idx2word = load_model(args.weights, args.vocab)
    print(f"\nGenerating caption for: {args.image}")
    caption = generate_caption(
        args.image,
        caption_model,
        word2idx,
        idx2word,
        #tokenizer,
        temperature=args.temperature,
    )
    print(f"\nGenerated caption: {caption}")

    if args.display:
        display_image_with_cap(args.image, caption)

if __name__ == "__main__":
    main()



