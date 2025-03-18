
package com.eye.eye_dectection;
import android.content.Context;
import android.graphics.Bitmap;


import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class TFLiteFishDetector {

    private static final String MODEL_PATH = "model.tflite";
    private static final int MODEL_INPUT_SIZE = 640;
    private Interpreter interpreter;

    public TFLiteFishDetector(Context context) throws IOException {
        Interpreter.Options options = new Interpreter.Options();
        options.setNumThreads(Runtime.getRuntime().availableProcessors());

        interpreter = new Interpreter(loadModelFile(context, MODEL_PATH), options);
    }

    /**
     * Load the TFLite model from assets.
     */
    private MappedByteBuffer loadModelFile(Context context, String modelPath) throws IOException {
        try (FileInputStream fis = new FileInputStream(context.getAssets().openFd(modelPath).getFileDescriptor());
             FileChannel fileChannel = fis.getChannel()) {

            long startOffset = context.getAssets().openFd(modelPath).getStartOffset();
            long declaredLength = context.getAssets().openFd(modelPath).getDeclaredLength();

            return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
        }
    }

    /**
     * Accepts OpenCV Mat directly, avoiding costly Bitmap conversion in the main class.
     */
    public float[] detectFish(Mat frame) {
        Bitmap bitmap = Bitmap.createBitmap(frame.cols(), frame.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(frame, bitmap);
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(bitmap, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, true);
        TensorImage image = new TensorImage(interpreter.getInputTensor(0).dataType());
        image.load(resizedBitmap);
        TensorBuffer outputBuffer = TensorBuffer.createFixedSize(
                interpreter.getOutputTensor(0).shape(),
                interpreter.getOutputTensor(0).dataType()
        );
        interpreter.run(image.getBuffer(), outputBuffer.getBuffer());

        return outputBuffer.getFloatArray();
    }

    /**
     * Close the interpreter to release resources.
     */
    public void close() {
        if (interpreter != null) {
            interpreter.close();
            interpreter = null;
        }
    }
}
