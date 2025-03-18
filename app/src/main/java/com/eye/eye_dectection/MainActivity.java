package com.eye.eye_dectection;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.util.Log;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.appcompat.app.ActionBar;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.TickMeter;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    private JavaCameraView cameraView;
    private TextView fpsTextView, accuracyTextView;
    private ImageButton back;
    private TFLiteFishDetector fishDetector;
    private int zoomLevel = 0;
    private Handler handler = new Handler();
    private TickMeter timer = new TickMeter();
    private ExecutorService executorService = Executors.newSingleThreadExecutor();

    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e("CountAct", "OpenCV initialization error");
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        ActionBar actionBar = getSupportActionBar();
        if(actionBar !=null){
            actionBar.hide();
        }
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, PackageManager.PERMISSION_GRANTED);
        fpsTextView = findViewById(R.id.fpsTextView);
        accuracyTextView = findViewById(R.id.accuracyTextView);
        back = findViewById(R.id.back);

        try {
            fishDetector = new TFLiteFishDetector(this);
        } catch (IOException e) {
            Log.e("TFLite", "Model loading failed", e);
            Toast.makeText(this, "Failed to load model", Toast.LENGTH_SHORT).show();
        }

        cameraView = findViewById(R.id.object);
        cameraView.setCvCameraViewListener(this);
        cameraView.setCameraIndex(JavaCameraView.CAMERA_ID_BACK);
        cameraView.enableView();

        back.setOnClickListener(v -> finish());

        ImageButton zoomInButton = findViewById(R.id.zoomInButton);
        ImageButton zoomOutButton = findViewById(R.id.zoomOutButton);

        zoomInButton.setOnClickListener(v -> zoomIn());
        zoomOutButton.setOnClickListener(v -> zoomOut());
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);
        requestCameraPermission();
    }

    private void requestCameraPermission() {
        if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{android.Manifest.permission.CAMERA}, 100);
        } else {
            cameraView.enableView();
        }
    }

    private void zoomIn() {
        if (cameraView != null) {
            zoomLevel += 1;
            cameraView.setZoom(zoomLevel);
        }
    }

    private void zoomOut() {
        if (cameraView != null && zoomLevel > 0) {
            zoomLevel -= 1;
            cameraView.setZoom(zoomLevel);
        }
    }

    @Override
    public void onCameraViewStarted(int width, int height) {}

    @Override
    public void onCameraViewStopped() {}

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat rgba = inputFrame.rgba();
        timer.start();

        executorService.execute(() -> {
            float[] results = fishDetector.detectFish(rgba);

            boolean isDetected = results[0] > results[1];
            float confidence = Math.max(results[0], results[1]) * 100;

            Log.d("EyeDectecion", String.format(Locale.getDefault(),
                    "Healthy: %.2f%%, Pink eye: %.2f%%, Detected: %s",
                    results[0] * 100, results[1] * 100, isDetected ? "Healthy" : "Pink eye"));

            runOnUiThread(() -> {
                String label = isDetected ? "Healthy" : "Pink eye";
                accuracyTextView.setText(String.format(label, confidence));

                if (isDetected) {
                    Rect box = new Rect(100, 150, 400, 500);
                    Scalar green = new Scalar(0, 255, 0);
                    int thickness = 4;
                    Imgproc.rectangle(rgba, box.tl(), box.br(), green, thickness);
                }
            });
        });

        timer.stop();
        double fps = 1.0 / timer.getTimeSec();
        runOnUiThread(() -> fpsTextView.setText(String.format(Locale.getDefault(), "FPS: %.2f", fps)));

        return rgba;
    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        handler.removeCallbacksAndMessages(null);
        if (cameraView != null) {
            cameraView.disableView();
        }
        if (fishDetector != null) {
            fishDetector.close();
        }
        executorService.shutdown();
    }
}
