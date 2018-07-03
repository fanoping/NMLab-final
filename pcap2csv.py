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
        cnt = 0
        tmp = os.popen('ifconfig').read()
        INTERFACE = tmp.split()[0]
        IP = os.popen('hostname -I').read().strip()
        kill_signal = False

        while not kill_signal:
            PCAP = "realtime{}.pcap".format(cnt)
            os.chdir(BIN)
            input_dir = os.path.join("../..", "Pcaps/realtime", PCAP)
            output_dir = os.path.join("../..", args.output)
            print('======  pcap name: {}  ======'.format(PCAP))
            print('IP address: ', IP)
            print('Interface: ', INTERFACE)
            print('# packets: ', args.packet_cnt)
        
            os.system("sudo tcpdump -c {} host {} -w {}".format(args.packet_cnt, IP, input_dir))
            os.system("./cfm {} {}".format(input_dir, output_dir))
            csv_name = args.output + PCAP + "_Flow.csv"
            cnt += 1
            os.chdir('../..')
            try:
                os.system("python3 classifierA.py -k 5 --test-csv {} ".format(csv_name))
                os.system("python3 classifierB.py -k 50 --test-csv {} ".format(csv_name))
            except KeyboardInterrupt:
                print('interrupted!')
                kill_signal = True

    # offtime
    else:
        print('=== offtime version ===')
        os.chdir(BIN)
        input_dir = os.path.join("../..", args.input)
        output_dir = os.path.join("../..", args.output)
        if os.path.isdir(input_dir):
            pcap_list = os.listdir(input_dir)
        else:
            _, pcap_name = os.path.split(args.input)
            pcap_list = [pcap_name]

        os.system("./cfm {} {}".format(input_dir, output_dir))
        for pcap in pcap_list:
            os.chdir('../..')
            csv_name = os.path.join(args.output, "{}_Flow.csv".format(pcap))
            print("==== csv directoy: {} ====".format(csv_name))
            # os.system("python3 classifierA.py -k 5 --test-csv {}".format(csv_name))
            os.system("python3 classifierB.py -k 50 --test-csv {}".format(csv_name))
            os.chdir(BIN)

if __name__ == "__main__":
    args = ArgumentParser()
    main(args)
