#include "WiFiEsp.h"

// Emulate Serial1 on pins 6/7 if not present
#ifndef HAVE_HWSERIAL1
#include "SoftwareSerial.h"
SoftwareSerial Serial1(6, 7); // RX, TX
#endif

#define PERIOD 2000000  // period in us
unsigned long last_us = 0L;
int x=0;
String sensor_val_str;
int status = WL_IDLE_STATUS;     // the Wifi radio's status
char ssid[] = "AndroidAP";            // your network SSID (name)
char pass[] = "sluz7956";        // your network password
char server[] = "192.168.43.186";
char led_arr[4];

// Initialize the Ethernet client object
WiFiEspClient client;

void setup()
{
  //initiate the I/O
  pinMode(2, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(3, OUTPUT);
  pinMode(8, OUTPUT);
  pinMode(9, OUTPUT);
  pinMode(10, OUTPUT);
  pinMode(11, OUTPUT);

  digitalWrite(5, LOW);
  delay(100);
  digitalWrite(5, HIGH);
  
  // initialize serial for debugging
  Serial.begin(115200);
  // initialize serial for ESP module
  Serial1.begin(9600);
  // initialize ESP module
  WiFi.init(&Serial1);

  // check for the presence of the shield
  while(WiFi.status() == WL_NO_SHIELD) {
    Serial.println("WiFi shield not present");
    WiFi.init(&Serial1);
    // don't continue
    //while (true);
  }

  digitalWrite(2, LOW); 
  digitalWrite(3, LOW); 
  // attempt to connect to WiFi network
  while ( status != WL_CONNECTED) {
    Serial.print("Attempting to connect to WPA SSID: ");
    Serial.println(ssid);
    // Connect to WPA/WPA2 network
    status = WiFi.begin(ssid, pass);
  }
  
  // you're connected now, so print out the data
  Serial.println("You're connected to the network");
  digitalWrite(2, HIGH); 
  
  printWifiStatus();

  Serial.println();
  Serial.println("Starting connection to server...");
  
}

void loop()
{
/////////////////////////////////////////////////////////////////////////////////////////////////////
  String content = "deviceidField=01234569&inputdataField=";
  for (int i = 0; i < 128; i++) {
    int sensorValue = analogRead(A0);
    if (sensorValue < 10){
      sensor_val_str = "000"+String(sensorValue);
    }
    else if(sensorValue < 100){
      sensor_val_str = "00"+String(sensorValue);
    }
    else if(sensorValue < 1000){
      sensor_val_str = "0"+String(sensorValue);
    }
    else{
      sensor_val_str = String(sensorValue);
    }
    content.concat(sensor_val_str);
    delay(15);
  }
/////////////////////////////////////////////////////////////////////////////////////////////////////
 
  if (client.connect(server, 3000)) {
    
    digitalWrite(2, HIGH); 

    digitalWrite(3, HIGH); 
    Serial.println("Connected to server");
    client.print("POST http://");
    client.print(server);
    client.println(":3000/injectDB HTTP/1.1");
    client.print("Host: ");
    client.print(server);
    client.println(":3000");
    client.println("Connection: keep-alive");
    client.println("Accept: */*");
    client.print("Content-Length: ");
    client.println(content.length());
    client.println("Content-Type: application/x-www-form-urlencoded; charset=UTF-8");
    client.println();
    client.println(content);
    client.println();

    }
  else{
    //digitalWrite(2, LOW); 
      //Toggle if it is not connected 
      if (x == 0) {
      // Toggle on
      digitalWrite(2, HIGH);
      x = 1;

    } else {
      // Toggle off
      digitalWrite(2, LOW);
      x = 0;
    }
    Serial.println("Not connected to server");
  }

    int k = 0;
    int led_idx = 0;
    int client_flag = 0;
    //last_us = 0L;
    //while(client_flag < 10){
      while (client.available()) {
        char c = client.read();
        Serial.print(c);

        if((k==0)&&(c == 'N')){
          k++;
        }
        else if(k==1){
          if(c == 'U')
            k++;
          else
            k=0;
        }
        else if(k==2){
          if(c == 'M')
            k++;
          else
            k=0;
        }
        else if(k==3){
          if(c == ':')
            k++;
          else
            k=0;
        }
        else if(k==4){
          led_arr[led_idx] = c;
          led_idx++;
          if(led_idx==4){
            client_flag = 2;
            if(led_arr[0]=='0')   
              digitalWrite(8, LOW);
            else if(led_arr[0]=='1')
              digitalWrite(8, HIGH);
            if(led_arr[1]=='0')   
              digitalWrite(9, LOW);
            else if(led_arr[1]=='1')
              digitalWrite(9, HIGH);
            if(led_arr[2]=='0')   
              digitalWrite(10, LOW);
            else if(led_arr[2]=='1')
              digitalWrite(10, HIGH);
            if(led_arr[3]=='0')   
              digitalWrite(11, LOW);
            else if(led_arr[3]=='1')
              digitalWrite(11, HIGH);
          } 
        }
      /*}
      if (micros () - last_us > PERIOD){
        last_us += PERIOD ;
        Serial.println(client_flag);
        client_flag ++;
      }*/
  }
  client.flush();
  client.stop(); 
  digitalWrite(3, LOW);


  }



void printWifiStatus()
{
  // print the SSID of the network you're attached to
  Serial.print("SSID: ");
  Serial.println(WiFi.SSID());

  // print your WiFi shield's IP address
  IPAddress ip = WiFi.localIP();
  Serial.print("IP Address: ");
  Serial.println(ip);

  // print the received signal strength
  long rssi = WiFi.RSSI();
  Serial.print("Signal strength (RSSI):");
  Serial.print(rssi);
  Serial.println(" dBm");
}
