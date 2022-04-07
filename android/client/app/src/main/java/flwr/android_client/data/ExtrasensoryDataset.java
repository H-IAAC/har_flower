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
    String[] allColumnNames;
    List<String[]> allDataXInput = null;
    List<String[]> allDataYInput = null;
    List<List<Float>> allXFloatArray;
    List<List<String>> allYStringArray;
    List<Valuedataset> listAllXY = new ArrayList<Valuedataset>();
    String TAG ="Extrasensory";
    public List<Valuedataset> getListAllXY() {
        return listAllXY;
    }
    public void setListAllXY(List<Valuedataset> listAllXY) {
        this.listAllXY = listAllXY;
    }
    public List<List<String>> getAllYStringArray() {
        return allYStringArray;
    }
    public List<List<Float>> getAllXFloatArray() {
        return allXFloatArray;
    }

    public void setAllYStringArray(List<List<String>> allYStringArray) {
        this.allYStringArray = allYStringArray;
    }
    public void setAllXFloatArray(List<List<Float>> allXFloatArray) {
        this.allXFloatArray = allXFloatArray;
    }

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
        extrasensoryGetX(context, fileX);
        extrasensoryGetY(context, fileY);
        try {
            getDataByCategories();
        }
        catch (Exception e) {
            Log.e(TAG,e.getMessage());
        }


    }


    public void extrasensoryGetX(Context context, String file) {
        try {
            InputStreamReader reader = new InputStreamReader(context.getAssets().open(file));
            CSVParser parser = new CSVParserBuilder().withSeparator(',').build();
            com.opencsv.CSVReader csvReader = new CSVReaderBuilder(reader)
                    .withCSVParser(parser)
                    .build();
            allDataXInput = csvReader.readAll();
            allColumnNames = allDataXInput.remove(0);
            allXFloatArray = new ArrayList<List<Float>>();
            for (String[] row1 : allDataXInput.subList(0, allDataXInput.size())) {
                List<Float> floatList = new ArrayList<>();
                for (int i = 0; i < row1.length; i++) {
                    try {
                        floatList.add(Float.valueOf(row1[i]));
                    } catch (NumberFormatException e) {
                        e.printStackTrace();
                        System.err.println(row1[i]);
                    }
                }
                  Log.d(TAG, String.valueOf(floatList.size()));
                allXFloatArray.add(floatList);
            }
            // Read all data at once
            csvReader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public void extrasensoryGetY(Context context, String file) {
        try {
            InputStreamReader reader = new InputStreamReader(context.getAssets().open(file));
            CSVParser parser = new CSVParserBuilder().withSeparator(',').build();
            com.opencsv.CSVReader csvReader = new CSVReaderBuilder(reader)
                    .withCSVParser(parser)
                    .build();
            allDataYInput = csvReader.readAll();
            allColumnNames = allDataYInput.remove(0);
            allYStringArray = new ArrayList<List<String>>();
            for (String[] row1 : allDataYInput.subList(0, allDataYInput.size())) {
                List<String> listStringY = new ArrayList<>();
                listStringY.add(row1[0]);
                for (int i = 1; i < row1.length; i++) {
                    try {
                        if (Float.valueOf(row1[i]) > 0) {

                            listStringY.add(String.valueOf(i));
                            listStringY.add(allColumnNames[i]);
                            // Log.d("ResultY ", row1[0] + " " + allColumnNames[i] + " " + String.valueOf(Float.valueOf(i)));
                        }
                    } catch (NumberFormatException e) {
                        e.printStackTrace();
                    }
                }
                allYStringArray.add(listStringY);
            }
            // Read all data at once
            csvReader.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    void getDataByCategories() {
        for (int i = 0; i < allXFloatArray.size(); i++) {
            try {
                if (labels.contains(String.valueOf(allYStringArray.get(i).get(2)))) {
                    Valuedataset d1 = new Valuedataset(allXFloatArray.get(1).get(0), String.valueOf(allYStringArray.get(i).get(1)), String.valueOf(allYStringArray.get(i).get(2)), allXFloatArray.get(i).subList(1, allXFloatArray.get(i).size()));
                    listAllXY.add(d1);
                }
            } catch (Exception e) {
            }
        }
    }

    public List<Valuedataset> getDataByCategory(String label) {
        List<Valuedataset> listcategoryXY = new ArrayList<Valuedataset>();
        for (int i = 0; i < listAllXY.size(); i++) {
            try {
                if (listAllXY.get(i).nameClass.contains(label)) {
                    listcategoryXY.add(listAllXY.get(i));
                }
            } catch (Exception e) {
                // Log.e("Logdata Erro1", String.valueOf(allXFloatArray.get(i).get(0)));
            }
        }
        return listcategoryXY;
    }

}
