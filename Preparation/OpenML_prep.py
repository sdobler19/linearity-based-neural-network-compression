import keras
import torch

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer


def train_validation_test_split(X, y, random_state = 1):
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size = 0.2, random_state = random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size = 0.25, random_state = random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_features(train_features, test_features):
    # Identify categorical and continuous columns
    categorical_features = train_features.select_dtypes(include=['object', 'category']).columns
    continuous_features = train_features.select_dtypes(include=['float64', 'int64']).columns

    # Create pipelines for continuous and categorical features
    continuous_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values for continuous data
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value="Missing Value")),  # Handle missing values for categorical data
        ('encoder', OneHotEncoder(sparse_output= False, handle_unknown='ignore', max_categories=20))  # Convert categorical data to one-hot vectors
    ])

    # Combine pipelines into a single ColumnTransformer
    preprocessor = ColumnTransformer([
        ('continuous', continuous_pipeline, continuous_features),
        ('categorical', categorical_pipeline, categorical_features)
    ])

    # Fit the preprocessor on the training data
    preprocessor.fit(train_features)

    # Transform both training and test data
    train_processed = preprocessor.transform(train_features)
    test_processed = preprocessor.transform(test_features)

    return train_processed, test_processed

def preprocess_labels(y_train, y_test):
    label_encoder = LabelEncoder()

    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    if len(label_encoder.classes_) > 2:
        onehot_encoder = OneHotEncoder(max_categories=20)
        y_train_encoded = y_train_encoded.reshape(-1, 1)
        y_test_encoded = y_test_encoded.reshape(-1, 1)
        y_train_encoded = onehot_encoder.fit_transform(y_train_encoded).todense()
        y_test_encoded = onehot_encoder.transform(y_test_encoded).todense()

    return y_train_encoded, y_test_encoded

def acc(pred, label):
    if len(label) > 1:
        return (torch.argmax(pred) == torch.argmax(label)).float()
    else:
        return (torch.argmax(pred) == label).float()
    
@torch.no_grad()
def eval_point(x, y, model, loss_func):
    pred = model(torch.as_tensor(x, dtype= torch.float32))
    loss = loss_func(pred, torch.as_tensor(y, dtype= torch.long)) # pred.dtype))# 
    return loss.item()

CeL = torch.nn.CrossEntropyLoss()

def evaluate(dataset, y, model, loss_func = CeL, acc = False):
    score = 0
    for idx in range(dataset.shape[0]):
        if acc:
            pred = model(torch.as_tensor(dataset.iloc[idx]))
            print(f"pred: {torch.argmax(pred)}, target: {torch.argmax(torch.as_tensor(y.iloc[idx], dtype= pred.dtype))}")
            score += (torch.argmax(pred) == torch.argmax(torch.as_tensor(y.iloc[idx], dtype= pred.dtype))).float().item()
        else:
            score += eval_point(dataset.iloc[idx], y.iloc[idx], model, loss_func)
    return score / dataset.shape[0]



def define_model(X_train, y_train, width_multiplyer = 1): 
   no_classes = 2
   if y_train.ndim > 1:  # Check if y_train_encoded is one-hot encoded
      no_classes = y_train.shape[1]
    
   # Define the Keras model
   model = keras.models.Sequential()
   model.add(keras.layers.Dense(int(64 * width_multiplyer), activation='relu', input_shape=(X_train.shape[1],)))
   # model.add(keras.layers.Dense(int(64 * width_multiplyer), activation='relu'))
   # model.add(keras.layers.Dense(int(128 * width_multiplyer), activation='relu'))
   # model.add(keras.layers.Dense(int(128 * width_multiplyer), activation='relu'))
   model.add(keras.layers.Dense(int(128 * width_multiplyer), activation='relu'))
   model.add(keras.layers.Dense(int(128 * width_multiplyer), activation='relu'))
   # model.add(keras.layers.Dense(int(256 * width_multiplyer), activation='relu'))
   # model.add(keras.layers.Dense(int(256 * width_multiplyer), activation='relu'))
   model.add(keras.layers.Dense(int(256 * width_multiplyer), activation='relu'))
   model.add(keras.layers.Dense(int(256 * width_multiplyer), activation='relu'))
   if no_classes == 2:
      model.add(keras.layers.Dense(1, activation='sigmoid')) 
      model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   else:
      model.add(keras.layers.Dense(no_classes, activation='softmax'))
      model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
   return model