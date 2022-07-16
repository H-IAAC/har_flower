package flwr.android_client.data;

import android.content.Context;
import android.util.Log;

import com.opencsv.CSVReader;

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.List;

import manifold.umap.Umap;

public class ExtrasensoryDataset {
    private static String[] labelsVAlues;
    private final Context context;
    private Umap umap = null;

    private float[][] data;
    String TAG = "Extrasensory";

    public static String[] getLabelsVAlues() {
        return labelsVAlues;
    }

    public static void setLabelsVAlues(String[] labelsVAlues) {
        ExtrasensoryDataset.labelsVAlues = labelsVAlues;
    }

    public float[][] getData() {
        return data;
    }

    public void setData(float[][] data) {
        this.data = data;
    }



    public float[][] getValues(String file, Boolean isTraining, boolean umapb){

        if (isTraining) {
            file = file + "train.csv";
            data = buildTable(this.context, file);
            if (umapb) {
                this.umap = new Umap();
                this.umap.setNumberComponents(5);         // number of dimensions in result
//                 umap.setNumberNearestNeighbours(10);
                this.umap.setThreads(1);
                this.umap.setVerbose(true);
                data = this.umap.fitTransform(data);

            }

        } else {
            file = file + "test.csv";
            data = buildTable(context, file);

            if (umapb) {
                data = this.umap.transform(data);
            }
        }


        Log.d(TAG, "manifold");


        Log.d("class ExtrasensoryDataset file: ", file + " size X: " + String.valueOf(data.length));
    return data    ;
    }
    public ExtrasensoryDataset(Context context) {
this.context=context;
    }


    public static float[][] buildTable(Context context, String csvPathFromResource) {
        try {
            InputStream is = context.getAssets().open(csvPathFromResource);
            InputStreamReader reader = new InputStreamReader(is, Charset.forName("UTF-8"));
            List<String[]> csv = new CSVReader(reader).readAll();
            csv.remove(0);
            labelsVAlues = new String[csv.size()];
            float[][] csv_double = new float[csv.size()][csv.get(0).length - 2];
            for (int i = 0; i < csv.size() - 1; i++) {
                for (int j = 1; j < csv.get(i).length - 1; j++) {
                    csv_double[i][j - 1] = Float.parseFloat(csv.get(i)[j]);
                }
                labelsVAlues[i] = csv.get(i)[227];
            }
            return csv_double;

        } catch (IOException e) {
            e.printStackTrace();
            return null;
        }
    }
}

