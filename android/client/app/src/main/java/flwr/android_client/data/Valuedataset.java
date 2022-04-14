package flwr.android_client.data;

import java.util.ArrayList;
import java.util.List;

public class Valuedataset {
    float line;
    String nameClass;
    String idClass;
    int quantity;
    List<Float> floatList = new ArrayList<>();

    public float getLine() {
        return line;
    }

    public String getNameClass() {
        return nameClass;
    }

    public void setNameClass(String nameClass) {
        this.nameClass = nameClass;
    }

    public String getIdClass() {
        return idClass;
    }

    public void setIdClass(String idClass) {
        this.idClass = idClass;
    }

    public int getQuantity() {
        return quantity;
    }

    public void setQuantity(int quantity) {
        this.quantity = quantity;
    }

    public List<Float> getFloatList() {
        return floatList;
    }

    public void setFloatList(List<Float> floatList) {
        this.floatList = floatList;
    }

    public void setLine(float line) {
        this.line = line;
    }

    public Valuedataset(float timestamp, String idClass, String nameClass, List<Float> floatList) {
        this.line = timestamp;
        this.nameClass = nameClass;
        this.floatList=floatList;
        this.idClass =idClass;
    }
}

