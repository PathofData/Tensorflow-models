# Tensorflow models
In this repository you can find Tensorflow-keras implementations of ResNeXt and dual path network model architectures.

### Instructions
In order to be able to use these models tensorflow version at least 2.1 needs to be installed on your environment.
Follow the instructions at https://www.tensorflow.org/install to see available options for installation.

In order to load the models onto your code follow the steps listed below.

Open a terminal and paste the following code to make a local copy of this repository:
```
# Clone the repository onto a local folder
git clone https://github.com/PathofData/Tensorflow-models.git
```

Then inside your python code import the model of your choice:

```
# Import the DPN module
from DPN50 import DPN50

# Initiallize a new model instance where the image dimension
# is (224, 224, 3) and we wish to predict 1000 classes
vision_model = DPN50(include_top=True,
                     weights=None,
                     input_tensor=None,
                     input_shape=(224, 224, 3),
                     pooling=None,
                     classes=1000)
```

Optionally print a summary of the model:
```
# Print each layere input, output shapes and number of parameters
vision_model.summary()
```

Once you have prepared a dataset for training use this model like how one would use any instance of keras models
```
# Compile the model with an optimizer and a loss function
vision_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Train the model on some data
vision_model.fit(X_train, y_train)
```
