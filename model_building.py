from tensorflow.keras import layers, models
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), padding = "same", activation='relu', input_shape = (48,48,1)))
model.add(layers.Conv2D(64, (3, 3), padding = "same",activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(128, (3, 3), padding = "same",activation='relu'))
model.add(layers.Conv2D(128, (3, 3), padding = "same",activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Conv2D(256, (3, 3), padding = "same",activation='relu'))
model.add(layers.Conv2D(256, (3, 3), padding = "same",activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(1000, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(lr=5e-5, rho=0.9, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])
history = model.fit(X_train,y_train, epochs=20, 
                    validation_data=(X_test, y_test))