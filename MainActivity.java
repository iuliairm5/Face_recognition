package com.example.facerecognition;

//import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_8U;
//import static org.bytedeco.javacpp.opencv_imgcodecs.cvLoadImage;
//import static org.bytedeco.javacpp.opencv_imgproc.cvCvtColor;

import static org.opencv.core.CvType.CV_32SC1;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_GRAYSCALE;
import static org.opencv.imgcodecs.Imgcodecs.IMREAD_UNCHANGED;
import static org.opencv.imgcodecs.Imgcodecs.imdecode;
import static org.opencv.imgcodecs.Imgcodecs.imread;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.content.Loader;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.os.Bundle;
import android.os.FileUtils;
import android.os.Looper;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.TextView;
import android.widget.Toast;

//import org.bytedeco.javacpp.opencv_core;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.face.EigenFaceRecognizer;
import org.opencv.face.FaceRecognizer;
import org.opencv.face.LBPHFaceRecognizer;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FilenameFilter;
import java.io.IOException;
import java.io.InputStream;
import java.sql.Array;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private static final String TAG = MainActivity.class.getSimpleName();
    static TextView tv,tv2;

    JavaCameraView javaCameraView;
    private CascadeClassifier faceDetector;
    //private EigenFaceRecognizer faceRecognizer000;
    FaceRecognizer faceRecognizer00;


    private LoaderCallbackInterface initCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");
                    InputStream inputStream = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                    File cascadeClassifier = getDir("cascade", Context.MODE_PRIVATE);
                    File lbpClassifier = new File(cascadeClassifier, "lbpcascade_frontalface.xml");
                    FileOutputStream fos = null;

                    try {
                        fos = new FileOutputStream(lbpClassifier);
                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = inputStream.read(buffer)) != -1) {
                            fos.write(buffer, 0, bytesRead);
                        }
                        inputStream.close();
                        fos.close();

                        faceDetector = new CascadeClassifier(lbpClassifier.getAbsolutePath());
                        //faceRecognizer000  = EigenFaceRecognizer.create();
                        //faceRecognizer00 = LBPHFaceRecognizer.create();
                    } catch (FileNotFoundException e) {
                        e.printStackTrace();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                    javaCameraView.enableView();
                    break;
                }
                default:
                    super.onManagerConnected(status);
            }
        }
    };
    private Mat matRGB;
    private Mat matGrey ;
    private int cameraId;

    TextView textView,textView3;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        javaCameraView = findViewById(R.id.javaCameraView);

        if(!OpenCVLoader.initDebug())
        {
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, initCallback);
        }
        else
            initCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);

        javaCameraView.setCvCameraViewListener(this);

        javaCameraView.setCameraPermissionGranted();

        textView = findViewById(R.id.textView4);
        textView3 = findViewById(R.id.textView3);

        javaCameraView.setCameraIndex(cameraId);

        javaCameraView.enableView();

        /////////////////////////////////////////////////// TRAINING ///////////////////////////////////////////

        faceRecognizer00 = LBPHFaceRecognizer.create();

        Mat dest = new Mat();
        Mat dest2 = new Mat();
        Mat dest3 = new Mat();
        Mat dest4 = new Mat();
        Mat dest5 = new Mat();
        Mat dest6 = new Mat();
        Mat dest7=new Mat();
        Mat dest8=new Mat();
        Mat dest9=new Mat();
        Mat dest10=new Mat();
        Mat dest11=new Mat();
        Mat dest12=new Mat();
        ArrayList<Mat> images = new ArrayList<>();
        Mat labels0 = new Mat (10,1,CV_32SC1);
        int ctr0 = 0;
        int label0 = 55;


        Bitmap btmp1 = BitmapFactory.decodeResource(getResources(),R.drawable.train1);
        Mat imgMat1 = convert(btmp1);
        //Mat imgMat1 = imread(f1.getAbsolutePath(), IMREAD_GRAYSCALE);
        //Mat imgMat1 = imread("1-jimcarrey_1.png");
        //img = cvLoadImage(f1.getAbsolutePath());
        //grayImg = opencv_core.IplImage.create(img.width(), img.height(), IPL_DEPTH_8U, 1);
        //cvCvtColor(img, grayImg, Imgproc.COLOR_RGB2GRAY);
        Imgproc.cvtColor(imgMat1, dest, Imgproc.COLOR_RGB2GRAY);
        images.add(ctr0,dest);
        labels0.put(ctr0,0,label0);
        ctr0++;


        label0 = 55;
        Bitmap btmp2 = BitmapFactory.decodeResource(getResources(),R.drawable.train2);
        Mat imgMat2 = convert(btmp2);
        Imgproc.cvtColor(imgMat2, dest2, Imgproc.COLOR_RGB2GRAY);
        images.add(ctr0,dest2);
        labels0.put(ctr0,0,label0);
        ctr0++;


        label0 = 55;
        Bitmap btmp3 = BitmapFactory.decodeResource(getResources(),R.drawable.train3);
        Mat imgMat3 = convert(btmp3);
        Imgproc.cvtColor(imgMat3, dest3, Imgproc.COLOR_RGB2GRAY);
        images.add(ctr0,dest3);
        labels0.put(ctr0,0,label0);
        ctr0++;

        label0 = 11;
        Bitmap btmp4 = BitmapFactory.decodeResource(getResources(),R.drawable.iuliatrain1);
        Mat imgMat4 = convert(btmp4);
        Imgproc.cvtColor(imgMat4, dest4, Imgproc.COLOR_RGB2GRAY);
        images.add(ctr0,dest4);
        labels0.put(ctr0,0,label0);
        ctr0++;

        label0 = 11;
        Bitmap btmp5 = BitmapFactory.decodeResource(getResources(),R.drawable.iuliatrain2);
        Mat imgMat5 = convert(btmp5);
        Imgproc.cvtColor(imgMat5, dest5, Imgproc.COLOR_RGB2GRAY);
        images.add(ctr0,dest5);
        labels0.put(ctr0,0,label0);
        ctr0++;

        label0=11;
        Bitmap btmp6 = BitmapFactory.decodeResource(getResources(),R.drawable.iuliatrain3);
        Mat imgMat6 = convert(btmp6);
        Imgproc.cvtColor(imgMat6, dest6, Imgproc.COLOR_RGB2GRAY);
        images.add(ctr0,dest6);
        labels0.put(ctr0,0,label0);
        ctr0++;

        label0=11;
        Bitmap btmp11 = BitmapFactory.decodeResource(getResources(),R.drawable.iuliatrain4);
        Mat imgMat11 = convert(btmp11);
        Imgproc.cvtColor(imgMat11, dest11, Imgproc.COLOR_RGB2GRAY);
        images.add(ctr0,dest11);
        labels0.put(ctr0,0,label0);
        ctr0++;

        label0=11;
        Bitmap btmp12 = BitmapFactory.decodeResource(getResources(),R.drawable.iuliatrain5);
        Mat imgMat12 = convert(btmp12);
        Imgproc.cvtColor(imgMat12, dest12, Imgproc.COLOR_RGB2GRAY);
        images.add(ctr0,dest12);
        labels0.put(ctr0,0,label0);
        ctr0++;

        label0 = 55;
        Bitmap btmp7 = BitmapFactory.decodeResource(getResources(),R.drawable.train6);
        Mat imgMat7 = convert(btmp7);
        Imgproc.cvtColor(imgMat7, dest7, Imgproc.COLOR_RGB2GRAY);
        images.add(ctr0,dest7);
        labels0.put(ctr0,0,label0);
        ctr0++;

        label0=55;
        Bitmap btmp8 = BitmapFactory.decodeResource(getResources(),R.drawable.train5);
        Mat imgMat8 = convert(btmp8);
        Imgproc.cvtColor(imgMat8, dest8, Imgproc.COLOR_RGB2GRAY);
        images.add(ctr0,dest8);
        labels0.put(ctr0,0,label0);
        ctr0++;

/*
        label0=3;
        Bitmap btmp9 = BitmapFactory.decodeResource(getResources(),R.drawable.genericgirl);
        Mat imgMat9 = convert(btmp9);
        Imgproc.cvtColor(imgMat9, dest9, Imgproc.COLOR_RGB2GRAY);
        images.add(ctr0,dest9);
        labels0.put(ctr0,0,label0);
        ctr0++;

        Bitmap btmp10 = BitmapFactory.decodeResource(getResources(),R.drawable.train7);
        Mat imgMat10 = convert(btmp10);
        Imgproc.cvtColor(imgMat10, dest10, Imgproc.COLOR_RGB2GRAY);
        images.add(ctr0,dest10);
        labels0.put(ctr0,0,label0);
        ctr0++;
*/


        faceRecognizer00.train(images, labels0);


    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        matRGB = new Mat();
        matGrey = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        matRGB.release();
        matGrey.release();
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        matRGB = inputFrame.rgba();
        matGrey = inputFrame.gray();
        MatOfRect faces = new MatOfRect();
        faceDetector.detectMultiScale(matRGB, faces);
        for(Rect rect: faces.toArray())
        {
            Imgproc.rectangle(matRGB, new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width, rect.y + rect.height),
                    new Scalar(255, 0, 0));


            ///////////////////////////////////////////////// PREDICTING ////////////////////////////////////////////
            int[] label0 = new int[1];
            double[] confidence = new double[1];


            faceRecognizer00.predict(matGrey, label0, confidence);


            //Bitmap btmptest = BitmapFactory.decodeResource(getResources(),R.drawable.test);
            //Mat imgMattest = convert(btmptest);
            //Mat destTest = new Mat();
            //Imgproc.cvtColor(imgMattest, destTest, Imgproc.COLOR_RGB2GRAY);
            //faceRecognizer00.predict(destTest, label0, confidence);
            int outputLabel = label0[0];
            //Toast.makeText(getApplicationContext(),"Predicted Label is", Toast.LENGTH_SHORT).show();
            //Toast.makeText(getApplicationContext(),predictedLabel, Toast.LENGTH_SHORT).show();


            textView.setText("Prediction says it is "+outputLabel+ " with a confidence of " + confidence[0]);
            if(outputLabel==55 & confidence[0] >40) textView3.setText("Oh, you are Jim Carrey !");
            else if (outputLabel==11 & confidence[0] >40) textView3.setText("It's just me, Irimia Iulia");

            if(confidence[0]<40) textView3.setText("I don't recognize this face. Who are you ? ");
        }


        return matRGB;
    }




    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.menu, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(@NonNull MenuItem item) {
        if(item.getItemId() == R.id.swap)
            swapCamera();
        return super.onOptionsItemSelected(item);
    }

    private void swapCamera() {
        cameraId = cameraId^1;
        javaCameraView.disableView();
        javaCameraView.setCameraIndex(cameraId);
        javaCameraView.enableView();
    }

    public Mat convert (Bitmap img)
    {
        Mat dest = new Mat();
        Utils. bitmapToMat( img, dest);
        return dest;
    }

}


