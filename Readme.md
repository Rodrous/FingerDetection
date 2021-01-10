# Finger Detection

# AIM

The Goal of this Minor Project is to detect and Count the number of fingers.

# Libraries Used

### OpenCv

OpenCV is a cross-platform library using which we can develop real-time computer vision applications. It mainly focuses on image processing, video capture and analysis including features like face detection and object detection.

### Numpy

NumPy is a Python library used for working with arrays. It also has functions for working in domain of linear algebra, fourier transform, and matrices.

NumPy was created in 2005 by Travis Oliphant. It is an open source project and you can use it freely. NumPy stands for Numerical Python.

## Process

1. Reading The Image and Converting it from RGB to HSV.

    We can do that with:

    ```python
    cv.cvtColor(image, cv.COLOR_BGR2HSV)
    ```

    - Math behind it:

        In case of 8-bit and 16-bit images, R, G, and B are converted to the floating-point format and scaled to fit the 0 to 1 range.

        Its fairly simple to understand, the Image array is inserted in the vector, max part of the array is getting in V and with the help of that we are computing other Hue and Saturation.

        ![Finger%20Detection%2067023c85a43f481ab8cbba5320e4aa30/Untitled.png](Finger%20Detection%2067023c85a43f481ab8cbba5320e4aa30/Untitled.png)

        The values are then converted to the destination data type:

        - 8-bit images: *`V*←255*V*,*S*←255*S*,*H*←*H*/2(to fit to 0 to 255)`
        - 16-bit images: (currently not supported): *`V*<−65535*V*,*S*<−65535*S*,*H*<−*H*`
        - 32-bit images: H, S, and V are left as is
2. Determining and highlighting the Color of Skin.

    Basically, it's hard to have one fixed color range for skin, because even if you want to detect only your own skin, its color will actually change a lot depending on lighting conditions.

    - I defined `Lower and Upper Values` of my **hand skin** with trail and error and still its not perfect
    - If you want to change that:

        ```python
        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([20, 255, 255], dtype = "uint8")
        ```

        This is the piece of Code you want to edit out

3. Finding the boundary of the image.
4. Creating a convex hull {outlines}.
    - The function [cv::convexHull](https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#ga014b28e56cb8854c0de4a211cb2be656) finds the convex hull of a 2D point set using the Sklansky's algorithm [[219]](https://docs.opencv.org/master/d0/de3/citelist.html#CITEREF_Sklansky82) that has *O(N logN)* complexity in the current implementation.

        ![Finger%20Detection%2067023c85a43f481ab8cbba5320e4aa30/Untitled%201.png](Finger%20Detection%2067023c85a43f481ab8cbba5320e4aa30/Untitled%201.png)

        ![Finger%20Detection%2067023c85a43f481ab8cbba5320e4aa30/Untitled%202.png](Finger%20Detection%2067023c85a43f481ab8cbba5320e4aa30/Untitled%202.png)

5. Finding Fingers Using the Cosine Theorem.
    - The Theorem simply says if the angle between two fingers is less than or equal to $89^o$ then its a finger otherwise its not

        ![Finger%20Detection%2067023c85a43f481ab8cbba5320e4aa30/Untitled%203.png](Finger%20Detection%2067023c85a43f481ab8cbba5320e4aa30/Untitled%203.png)

        ![Finger%20Detection%2067023c85a43f481ab8cbba5320e4aa30/Untitled%204.png](Finger%20Detection%2067023c85a43f481ab8cbba5320e4aa30/Untitled%204.png)

6. Finding the defects and removing them.

    It is still not perfect but it should take only your hand as a Region of Interest

    [Contours : More Functions - OpenCV-Python Tutorials 1 documentation](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html)

7. Counting the fingers.

## USAGE

```python
python ./Finger_counting.py
```

## FURTHER DEVELOPMENTS

I've trained a TensorFlow model on **Naruto Handsigns** and added to the `/bin` file, the idea is to add a gesture detection system which will Determine what handsigns you are making.

I've added a seprate bit of code on how the images will be read and how will they be processed before giving out the prediction.

I didn't completed it yet but I will do that bit soon, when I have time and energy to spare.

## KNOWN ISSUES

- `OOPModel` is taking too much time to run/not executing it.