import xmlrpc.client
import time

with xmlrpc.client.ServerProxy('http://localhost:8000/', allow_none=True) as client:
    client.add_to_scale(-80)