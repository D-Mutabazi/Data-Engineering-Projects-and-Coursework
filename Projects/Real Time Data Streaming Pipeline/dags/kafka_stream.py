from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator   #operator to fetch data


#default arg -to attach to DAG
default_args ={
    'owner':'airscholar',
    'start_date': datetime(2024,4,3, 0,0 ),
    'timezone': 'UTC'
}

def stream_data():
    import json
    import requests

    resp = requests.get('https://randomuser.me/api/')
    resp = resp.json()
    resp = resp['results'][0]
    print(json.dumps(resp, indent=3))  # structures JSON data to make it easily readible

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