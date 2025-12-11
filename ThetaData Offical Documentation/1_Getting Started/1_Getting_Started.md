What is Theta Terminal and why do I need it?
Theta Data uses a proprietary protocol to send you data. This allows for up to a 30x reduction in bandwidth. Theta Terminal hosts a local HTTP server on your machine that only you can access. The REST API endpoints in this documentation connect to that server. Theta Terminal is a value-add: it reduces latency, increases throughput, and improves overall market data delivery performance.

REQUIRED

The Theta Terminal must be running to access data; this documentation is for v3 only!

ThetaTerminal.drawio.png

Installing Java
Java 21 or higher is required to use the Theta Terminal.

Recommendation

Use Java 21 or higher as performance for optimal performance and stability.

Check Java Version
You can easily check your Java version by typing the following in your command prompt / terminal:


java -version
If you encounter a message such as java is an unregonized command, you need to install Java. If your Java version is under 21.0.0, you need to install a newer version of Java.

image.png

Install Java on Windows & Mac
Download & open the Java installer on the Oracle Website and run the installer. Type java -version again to verify the minimum required version of Java has been installed.

Install Java on Ubuntu
Enter the following command in your terminal. You might require elevated privillages.


apt install openjdk-21-jdk openjdk-21-jre
Type java -version again to verify the minimum required version of Java has been installed.

Installing Theta Terminal

The Theta Terminal is an auto-updating Java JAR file. You do not need to "install" anything other than Java 21 or greater to get it to work. After downloading the JAR file, place it in a directory, we recommend ThetaTerminal. Then run the JAR file by issuing the following command:


java -jar ThetaTerminalv3.jar
That is it! The Theta Terminal will ensure you are running the latest version.

You can download the Theta Terminal here: ThetaTerminalv3.jar

Worried about downloading the terminal?
Jar files can be decompiled. This means you can view the source code of the Theta Terminal, so you can figure out what exactly is going on if you need to. We aim to be as transparent as possible with our software and data. If you still have questions, feel free to reach out to us so we can provide more information.

Worried about auto-upgrading?
The terminal will never attempt to update itself while running, only on startup. The terminal will also fall-back to a previous version if for some reason the new version fails to run on your system. To avoid automatic updates, you can run the library JAR directly by accessing it in the lib directory, located in the same directory as the ThetaTerminalv3.jar file.

Saving Your Creds
Before launching the terminal, your credentials (email and password) must be saved in a creds.txt file in the same directory as the ThetaTerminalv3.jar file. You can also specify an alternate location using the --creds-file command-line option.

To create a creds file, place your email address on the first line, and your password on the second line.

Config File
The only file required to run the Theta Terminal is a creds.txt file which holds your credentials.

When you run the terminal for the first time, it will write a default configuration file to the same location as the ThetaTerminalv3.jar file. For most use cases, you do not need to modify this file at all. There are 4 configs you might want to change:

host - This is the "host" the HTTP server will bind to. If you don't know what this means, don't change it 😃
port - This is the port the HTTP server will use. If you don't know what this means, don't change it 😃
log_directory - This is where logs for the terminal are stored.
request_queue_length - This is the number of requests to allow waiting to be executed when making concurrent requests.
Launch Theta Terminal
On the first run, the terminal will save a copy of the default configuration file. You can modify settings in this file, but for most use cases, you do not need to change anything.

The Theta Terminal can be run from the command line on Windows, Mac, and Linux by issuing the following command:


java -jar ThetaTerminalv3.jar
Creds File Required

Before launching the terminal, you must create a creds file with your email on the first line, and password on the second. Save the file to creds.txt, and place it in the same directory as ThetaTerminalv3.jar.

Good To Knows
If you have issues getting started, join our discord server and ask for help.

Does a datapoint look different from what you expected? This article contains information regarding the nuances of datasets available on this platform.

If you are making concurrent requests, follow the outlines provided in our concurrent requests article.

What's Next?
At this point the Theta Terminal should be running. If you encountered any issues, please reach out to us!

OpenAPI YAML
The OpenAPI YAML file is available for download here