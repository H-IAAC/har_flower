package flwr.android_client.data;

import android.content.Context;
import android.util.Log;

import com.opencsv.CSVParser;
import com.opencsv.CSVParserBuilder;
import com.opencsv.CSVReaderBuilder;

import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class ExtrasensoryDataset {
    private final List<String> labels;
    private String[] allColumnNamesY;



    List<Valuedataset> listAllXY = new ArrayList<Valuedataset>();
    String TAG = "Extrasensory";


    public ExtrasensoryDataset(Context context, String file, Boolean isTraining, List<String> labels) {
        this.labels = labels;
        String fileX;
        String fileY;

        if (isTraining) {
            fileX = file + "x_train.csv";
            fileY = file + "y_train.csv";

        } else {
            fileX = file + "x_test.csv";
            fileY = file + "y_test.csv";
        }
        List<String[]> dataX = extrasensoryGetData(context, fileX);
        List<String[]> datay = extrasensoryGetData(context, fileY);
        Log.d("class ExtrasensoryDataset file: ", file + " size X: " + String.valueOf(dataX.size()) + " size Y: " + String.valueOf(datay.size()));

        mixValues(dataX,datay);
    }


    public List<String[]> extrasensoryGetData(Context context, String file) {
         List<String[]> allDataInput = null;

        try {
            InputStreamReader reader = new InputStreamReader(context.getAssets().open(file));
            CSVParser parser = new CSVParserBuilder().withSeparator(',').build();
            com.opencsv.CSVReader csvReader = new CSVReaderBuilder(reader)
                    .withCSVParser(parser)
                    .build();
            allDataInput = csvReader.readAll();
            allColumnNamesY = allDataInput.remove(0);
            reader.close();
            csvReader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
return allDataInput;
    }


    public List<Valuedataset> getDataByCategory(String label) {
        List<Valuedataset> listcategoryXY = new ArrayList<Valuedataset>();
        for (int i = 0; i < listAllXY.size(); i++) {

            if (listAllXY.get(i).nameClass.contentEquals(label)) {
                listcategoryXY.add(listAllXY.get(i));
            }

        }
        return listcategoryXY;
    }


    void mixValues(List<String[]> dataX,List<String[]> dataY) {
               for (int i = 0; i < dataX.size(); i++) {
            List<Float> floatvalues = getFloatValues(dataX.get(i));
            getLabels(dataY.get(i), floatvalues);


        }
    }

    List<Float> getFloatValues(String[] strings) {
        List<Float> floatList = new ArrayList<>();
        for (int i = 0; i < strings.length; i++) {
            floatList.add(Float.valueOf(strings[i]));
        }
        return floatList;
    }

    void getLabels(String[] strings, List<Float> floatvalues) {
        for (int i = 0; i < strings.length; i++) {
            if (strings[i].contains("1.0")) {
                Valuedataset d1 = new Valuedataset(floatvalues.get(0), String.valueOf(i), allColumnNamesY[i], floatvalues.subList(1, floatvalues.size()));
                this.listAllXY.add(d1);
            }
        }

    }
}

