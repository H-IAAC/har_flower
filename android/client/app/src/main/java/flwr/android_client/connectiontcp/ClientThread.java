package flwr.android_client.connectiontcp;

import java.io.IOException;
import java.io.PrintWriter;
import java.net.Socket;


// the ClientThread class performs
// the networking operations
public class ClientThread implements Runnable {
    private final String message;
    private Socket client;
    private PrintWriter printwriter;
    private static final int TCP_SERVER_PORT = 9999;//should be same to the server port
    public ClientThread(String message) {
        this.message = message;
    }
    @Override
    public void run() {
        try {
            // the IP and port should be correct to have a connection established
            // Creates a stream socket and connects it to the specified port number on the named host.
            client = new Socket("192.168.15.90", 9999);  // connect to server
            printwriter = new PrintWriter(client.getOutputStream(),true);
            printwriter.write(message);  // write the message to output stream

            printwriter.flush();
            printwriter.close();

            // closing the connection
            client.close();

        } catch (IOException e) {
            e.printStackTrace();
        }

       /* // updating the UI
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                textField.setText("");
            }
        });*/
    }
}

    //replace runTcpClient() at onCreate with this method if you want to run tcp client as a service

