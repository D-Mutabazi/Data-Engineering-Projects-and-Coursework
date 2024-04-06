from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator   #operator to fetch data


#default arg -to attach to DAG
default_args ={
    'owner':'airscholar',
    'start_date': datetime(2024,4,3, 0,0 ),
    'timezone': 'UTC'
}

def get_Data():
    import requests

    resp = requests.get('https://randomuser.me/api/')
    resp = resp.json()
    resp = resp['results'][0]

    return resp


# formart data revcd rfrom API
def format_data(res):
    data = {} #dictionary
    #extract data from JSON object
    location = res['location']
    data['first_name'] = res['name']['first']
    data['last_name'] = res['name']['last']
    data['gender'] = res['gender']
    data['address'] = f"{str(location['street']['number'])} {location['street']['name']}, "\
                     f"{location['city']},{location['state']},{location['country']}"
    data['post_code'] = location['postcode']
    data['email'] = res['email']
    data['username'] = res['login']['username']
    data['dob'] = res['dob']['date']
    data['registered_date'] = res['registered']['date']
    data['phone'] = res['phone']
    data['picture'] = res['picture']['medium']

    return data



def stream_data():
    import json
    res = get_Data()
    res = format_data(res)
    print(json.dumps(res, indent=3))

#entry point
# with DAG('user_automation', 
#          default_args = default_args,
#          schedule_interval = '@daily',
#          catchup = False) as dag :
    
#     streaming_task = PythonOperator(
#         task_id = 'stream_data_from_api',
#         python_callback =stream_data
#     )

stream_data()