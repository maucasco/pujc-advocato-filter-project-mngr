Layer(type);OutputShape,Param#
conv2d_3(Conv2D);(None,498,498,32);896

max_pooling2d_3(MaxPooling2D);(None,249,249,32);0


conv2d_4(Conv2D);(None,245,247,64);30784

max_pooling2d_4(MaxPooling2D);(None,122,123,64);0


conv2d_5(Conv2D);(None,120,121,128);73856

max_pooling2d_5(MaxPooling2D);(None,60,60,128);0


flatten_1(Flatten);(None,460800);0

dense_2(Dense);(None,128);58982528

dropout_1(Dropout);(None,128);0

dense_3(Dense);(None,2);258
