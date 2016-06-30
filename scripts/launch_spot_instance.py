

from boto import ec2
from aws_config import *

import datetime
import time

conn = ec2.connect_to_region('us-east-1', aws_access_key_id = AWS_ACCESS_KEY,\
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

sg = 'sg-8ed5a8eb'
subnet = {'us-east-1b': 'subnet-347a7e1c',
          'us-east-1c': 'subnet-6255ae15',
          'us-east-1d': 'subnet-e8b551b1'}

ami = 'ami-24d01e49'

key_name = 'facebook'
def get_spot_prices(instance_type = 'c4.8xlarge'):
    s = datetime.datetime.now() - datetime.timedelta(hours = 1)
    prices = conn.get_spot_price_history(start_time = s.isoformat(),\
            end_time = datetime.datetime.now().isoformat(), instance_type = instance_type,\
            product_description='Linux/UNIX (Amazon VPC)')
    prices = filter(lambda x: x.availability_zone != 'us-east-1e', prices)
    sorted_prices = sorted(prices, cmp = lambda x, y: cmp(x.price, y.price))
    for x in sorted_prices:
        print x.region, x.availability_zone, x.price
    return sorted_prices

def request_spot(price = '0.9', instance_type = 'm4.large', av_zone = 'us-east-1b', name = 'launched from boto'):
    req = conn.request_spot_instances(price = price, count = 1,\
            image_id = ami, key_name = key_name, security_group_ids = [sg], \
            instance_type = instance_type, subnet_id = subnet[av_zone])
    spot_id = req[0].id
    req[0].add_tag('Name', name)
    print "requested spot", spot_id
    print "getting instance details"
    while True:
        time.sleep(30)
        sp_r = get_spot_instance(spot_id)
        if sp_r.state == 'active':
            time.sleep(30)
            ins_res = conn.get_all_instances(instance_ids = [sp_r.instance_id])
            instance = ins_res[0].instances[0]
            print "launched instance id", instance.id
            print "public dns", instance.public_dns_name
            instance.add_tag('Name', name)
            break
    return req[0]

def get_spot_instance(spot_id):
    ins = conn.get_all_spot_instance_requests(spot_id)
    return ins[0]

def get_instance(instance_id):
    instance = conn.get_all_instances(instance_ids = [sp_r.instance_id])[0].instances[0]
    return instance
