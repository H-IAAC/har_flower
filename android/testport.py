import socket
hostip = "192.168.15.90" 
print('host ip', hostip)# Should be displayed as: 127.0.1.1  
port = 9999     # Arbitrary non-privileged port  
flowerport=8080


def test_port(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((ip, port))
    sock.close()

    if result == 0:
        return True
    else:
        return False
        
print(test_port(hostip,flowerport))
