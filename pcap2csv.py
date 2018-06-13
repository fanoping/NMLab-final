import argparse
import os

def ArgumentParser():
    parser = argparse.ArgumentParser(description='=== internet flow detection ===')
    parser.add_argument('action', choices=['offtime', 'realtime'])
    parser.add_argument('--packet_cnt', default=50, type=int, 
                        help='number of packets to detect')
    parser.add_argument('--input', default='Pcaps/tor/', type=str, 
                        help='offtime pcap directory')
    parser.add_argument('--output', default='CSV/Scenario-C/', type=str, 
                        help='output csv directory')
    return parser.parse_args()


def main(args):
    BIN = './CICFlowMeter-4.0/bin'
    # realtime
    if args.action == 'realtime':
        print('=== realtime version ===')
        tmp = os.popen('ifconfig').read()
        interface = tmp.split()[0]
        IPaddress = os.popen('hostname -I').read().strip()
        while True:
            print("sudo tcpdump -i {} -c {} host {} -w realtime.pcap ".format(interface,
                                                                            args.packet_cnt, IPaddress))
            os.system("sudo tcpdump  -c {} host {} -w realtime.pcap ".format( args.packet_cnt, IPaddress))
            os.chdir(BIN)
            input_dir = os.path.join("../..", "realtime.pcap")
            output_dir = os.path.join("../..", args.output)
            csv_name = args.output + "realtime.pcap" + "_Flow.csv"
            os.system("./cfm {} {}".format(input_dir, output_dir))

    # offtime
    else:
        print('=== offtime version ===')
        os.chdir(BIN)
        input_dir = os.path.join("../..", args.input)
        output_dir = os.path.join("../..", args.output)
        pcap_name = args.input[args.input.rfind('/')+1:]
        csv_name = args.output + pcap_name + "_Flow.csv"
        print("csv_name: ", csv_name)
        os.system("./cfm {} {}".format(input_dir, output_dir))
    
    # pcap to csv 
    # os.chdir('../..')
    # os.system("python3 classifierA.py -k 10 --test-csv {} ".format(csv_name))
    # os.system("python3 classifierB.py -k 10 --test-csv {} ".format(csv_name))

if __name__ == "__main__":
    args = ArgumentParser()
    main(args)
