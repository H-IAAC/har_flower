package flwr.android_client;

import android.content.Context;
import android.os.ConditionVariable;
import android.util.Log;
import android.util.Pair;

import androidx.lifecycle.MutableLiveData;

import org.apache.commons.lang3.ArrayUtils;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;

import flwr.android_client.connectiontcp.ClientThread;
import flwr.android_client.data.ExtrasensoryDataset;
import flwr.android_client.data.Valuedataset;

public class FlowerClient {

    private TransferLearningModelWrapper tlModel;
    private static final int LOWER_BYTE_MASK = 0xFF;
    private MutableLiveData<Float> lastLoss = new MutableLiveData<>();
    private Context context;
    private final ConditionVariable isTraining = new ConditionVariable();
    private static String TAG = "Flower";
    private int local_epochs = 1;

    public FlowerClient(Context context) {
        this.tlModel = new TransferLearningModelWrapper(context);
        this.context = context;
    }

    public ByteBuffer[] getWeights() {
        return tlModel.getParameters();
    }

    public Pair<ByteBuffer[], Integer> fit(ByteBuffer[] weights, int epochs) {

        this.local_epochs = epochs;
        tlModel.updateParameters(weights);
        isTraining.close();
        tlModel.train(this.local_epochs);
        tlModel.enableTraining((epoch, loss) -> setLastLoss(epoch, loss));
        Log.e(TAG ,  "Training enabled. Local Epochs = " + this.local_epochs);
        isTraining.block();
        return Pair.create(getWeights(), tlModel.getSize_Training());
    }

    public Pair<Pair<Float, Float>, Integer> evaluate(ByteBuffer[] weights) {
        tlModel.updateParameters(weights);
        tlModel.disableTraining();
        return Pair.create(tlModel.calculateTestStatistics(), tlModel.getSize_Testing());
    }

    public void setLastLoss(int epoch, float newLoss) {
        if (epoch == this.local_epochs - 1) {
            Log.e(TAG, "Training finished after epoch = " + epoch);
            lastLoss.postValue(newLoss);
            tlModel.disableTraining();
            isTraining.open();
        }
    }

    /*public void loadData(int device_id) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/partition_" + (device_id - 1) + "_train.txt")));
            String line;
            int i = 0;
            while ((line = reader.readLine()) != null) {
                i++;
                Log.e(TAG, i + "th training image loaded");
                addSample("data/" + line, true);
            }
            reader.close();

            i = 0;
            reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/partition_" +  (device_id - 1)  + "_test.txt")));
            while ((line = reader.readLine()) != null) {
                i++;
                Log.e(TAG, i + "th test image loaded");
                addSample("data/" + line, false);
            }
            reader.close();

        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }
*/
    public void loadDataExtrasensory(int device_id, String experimentid) {

//        new Thread(new ClientThread("start")).start();
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/extrasensory/fold_" + experimentid + "/extrasensory_partition_" + (device_id - 1) + ".txt")));
            String line;
            int i = 0;
            while ((line = reader.readLine()) != null) {
                i++;
                loadDataExtrasensoryByClient("data/extrasensory" +  "/" + line, true);
            }
            reader.close();

            i = 0;
            reader = new BufferedReader(new InputStreamReader(this.context.getAssets().open("data/extrasensory/fold_" + experimentid + "/extrasensory_partition_" + (device_id - 1) + ".txt")));
            while ((line = reader.readLine()) != null) {
                i++;
                loadDataExtrasensoryByClient("data/extrasensory"  + "/" + line, false);
            }
            reader.close();

        } catch (IOException ex) {
            ex.printStackTrace();
        }
    }


    private void loadDataExtrasensoryByClient(String file, Boolean isTraining) throws IOException {
        // "label:LYING_DOWN","label:SITTING","label:FIX_walking",
        List<String> labels=  Arrays.asList( "label:LYING_DOWN","label:SITTING","label:FIX_walking","label:FIX_running","label:BICYCLING","label:SLEEPING");

        ExtrasensoryDataset extrasensoryDataset = new ExtrasensoryDataset(this.context, file, isTraining,labels);


        for (String label :labels) {
            List<Valuedataset> valuesCat = extrasensoryDataset.getDataByCategory(label);
            for (Valuedataset valueD :valuesCat){

                float[] floatArray1 = ArrayUtils.toPrimitive(valueD.getFloatList().toArray(new Float[0]), 0.0F);


                try {
                    this.tlModel.addSample(floatArray1, valueD.getNameClass(), isTraining).get();
                } catch (ExecutionException e) {
                    throw new RuntimeException("Failed to add sample to model", e.getCause());
                } catch (InterruptedException e) {
                    // no-op
                }}
}
        Log.d(TAG, "SAMPLE TRAIN  "+String.valueOf( this.tlModel.getSize_Training()) +" SAMPLE TEST:"+ String.valueOf(this.tlModel.getSize_Testing()));



    }}



   /* private void addSample(String photoPath, Boolean isTraining) throws IOException {
        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        Bitmap bitmap =  BitmapFactory.decodeStream(this.context.getAssets().open(photoPath), null, options);
        String sampleClass = get_class(photoPath);

        // get rgb equivalent and class
        float[] rgbImage = prepareImage(bitmap);

        // add to the list.
        try {
            this.tlModel.addSample(rgbImage, sampleClass, isTraining).get();
        } catch (ExecutionException e) {
            throw new RuntimeException("Failed to add sample to model", e.getCause());
        } catch (InterruptedException e) {
            // no-op
        }
    }

    public String get_class(String path) {
        String label = path.split("/")[2];
        return label;
    }

    *//**
     * Normalizes a camera image to [0; 1], cropping it
     * to size expected by the model and adjusting for camera rotation.
     *//*
    private static float[] prepareImage(Bitmap bitmap)  {
        int modelImageSize = TransferLearningModelWrapper.IMAGE_SIZE;

        float[] normalizedRgb = new float[modelImageSize * modelImageSize * 3];
        int nextIdx = 0;
        for (int y = 0; y < modelImageSize; y++) {
            for (int x = 0; x < modelImageSize; x++) {
                int rgb = bitmap.getPixel(x, y);

                float r = ((rgb >> 16) & LOWER_BYTE_MASK) * (1 / 255.0f);
                float g = ((rgb >> 8) & LOWER_BYTE_MASK) * (1 / 255.0f);
                float b = (rgb & LOWER_BYTE_MASK) * (1 / 255.0f);

                normalizedRgb[nextIdx++] = r;
                normalizedRgb[nextIdx++] = g;
                normalizedRgb[nextIdx++] = b;
            }
        }

        return normalizedRgb;
    }
}*/