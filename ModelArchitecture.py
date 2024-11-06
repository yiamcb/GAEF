from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, LSTM, GRU, Attention, GlobalMaxPooling1D, Concatenate, Dense, Dropout, Add
from keras.regularizers import l2

def create_model(input_shape, num_classes, dropout_rate=0.5, l2_penalty=0.01):
    # Input layer
    input_layer = Input(shape=input_shape)

    # CNN layers
    cnn_output = Conv1D(64, 3, activation='relu', padding='same')(input_layer)
    cnn_output = MaxPooling1D(2)(cnn_output)
    cnn_output = Conv1D(128, 3, activation='relu', padding='same')(cnn_output)
    cnn_output = MaxPooling1D(2)(cnn_output)
    cnn_output = Flatten()(cnn_output)

    # GRU layers
    gru_output = GRU(64, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)(input_layer)
    gru_output = GRU(64, return_sequences=True, dropout=dropout_rate, recurrent_dropout=dropout_rate)(gru_output)
    gru_output = Attention()([gru_output, gru_output])
    gru_output = GlobalMaxPooling1D()(gru_output)

    # Concatenate CNN and GRU outputs with skip connection
    combined_output = Concatenate()([cnn_output, gru_output])

    # Fully connected layers with regularization
    fc_output = Dense(128, activation='relu', kernel_regularizer=l2(l2_penalty))(combined_output)
    fc_output = Dropout(dropout_rate)(fc_output)
    output_layer = Dense(num_classes, activation='softmax')(fc_output)

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer)

    return model