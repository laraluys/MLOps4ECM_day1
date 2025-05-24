# Data labeling

When training a machine learning model you need data and when the model you want to train is a supervised machine learning model like a deep neural network, you also need labels for this data. Sometimes this labeling can be easy, like putting all the dog pictures in a folder called "dog" and all your cat pictures in a folder called "cat". However in most cases this is not the case. For example in the warp-D dataset. [This dataset](https://www.kaggle.com/datasets/parohod/warp-waste-recycling-plant-dataset) shows images of a waste recycling plant. The goal will be to detect certain types of bottles and plastic packages. However we do not want to only detect if a certain bottle is in the image, we also want to detect where it is. To do this we need to draw object detection boxes on the images. These can than be used to train the machine learning model. An example of these labels can be seen in the image below.

<img src="images/object_detection_example.webp" />

As you can see we have both a box drawn around certain objects and a label that corresponds with those boxes. The boxes are then saved by their top left and bottom rigth corners. However this is not easy to label by yourself. Therefore you can use tools like the open-source tool [label-studio](https://labelstud.io/). This tool helps making the labeling proces already a little easier. For this exercise you will label the images in the datasets/warp-D forlder. First you will need to start up the label-studio python library in the command line with the following command. 
</br>
<code>
label-studio start
</code>
</br>

This will open up a HTML-page on the local-host that looks like this.
<img src="images/label-studio_login.png" />
Here you will have to create an account and log in. When you have done that, you will arrive at the following page.
<img src="images/label-studio_welcome.png" />

From this point you can create a new project which you give a name and a discription.
<img src="images/label-studio_project_creation.png" />

Next, you can upload your data into your project by dragging and dropping or adding straight from your laptop.
<img src="images/label-studio_upload-data.png" />

The last step is to choose the type of labels you will need from the label-studio templates, or start from scratch and then set up the label names needed and other parameters of your labeling setup.
<img src="images/label-studio_setup-labeling.png" />
<img src="images/label-studio_setup-labeling_2.png" />

The labels you will need are:
* bottle-blue
* bottle-green
* bottle-dark
* bottle-milk
* bottle-transp
* bottle-multicolor
* bottle-yogurt
* bottle-oil
* cans
* juice-cardboard
* milk-cardboard
* detergent-color
* detergent-transparent
* detergent-box
* canister
* glass-transp
* glass-dark
* glass-green
* detergent-white

With all the labels set, you can start your project and arrive at the following page.
<img src="images/label-studio_labeling_start.png" />

From there you can just push the label all tasks and start labeling. Just draw the boxes on the image. Label-studio will do the rest.
<img src="images/label-studio_labeling_1.png" />

After labeling your images you will see that your labeled images have a different status in the project page. If you close label-studio and later come back, label-studio will have saved your previous labels as well.
<img src="images/label-studio_labeling_2.png" />

Then, to use your labels, you can export them (right top corner) and choose the way your labels should be represented. This depends on the type of model you want to train and which type of labels you have created.
<img src="images/label-studio_labeling_3.png" />

Sometimes, if there is a lot of data that needs to be labeled, you can train a machine learning model on the data that has been labeled and use it to label the data that does not have a label yet. You can also use the predictions or confidences of a machine learning model to decide which image should be labeled next. This can also be done with label-studio, if you go to the settings of your labeling project.

<img src="images/label-studio_labeling_extra.png" />