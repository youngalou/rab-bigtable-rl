import struct
import datetime

from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import bigtable
from google.cloud.bigtable import row_filters

def gcs_load_bucket(gcp_project_id, bucket_id, credentials):
    print('-> Looking for the [{}] bucket.'.format(bucket_id))
    storage_client = storage.Client(gcp_project_id, credentials=credentials)
    gcs_bucket = storage_client.get_bucket(bucket_id)
    print('-> Bucket found.')
    return gcs_bucket

def cbt_load_table(gcp_project_id, cbt_instance_id, cbt_table_name, credentials):
    print('-> Looking for the [{}] table.'.format(cbt_table_name))
    client = bigtable.Client(gcp_project_id, admin=True, credentials=credentials)
    instance = client.instance(cbt_instance_id)
    cbt_table = instance.table(cbt_table_name)
    if not cbt_table.exists():
        print("-> Table doesn't exist. Creating [{}] table...".format(cbt_table_name))
        max_versions_rule = bigtable.column_family.MaxVersionsGCRule(1)
        column_families = {'trajectory': max_versions_rule, 'global': max_versions_rule}
        cbt_table.create(column_families=column_families)
        print('-> Table created. Give it ~60 seconds to initialize before loading data.')
        exit()
    else:
        print("-> Table found.")
    return cbt_table

def gcp_load_pipeline(gcp_project_id, cbt_instance_id, cbt_table_name, bucket_id, credentials):
    cbt_table = cbt_load_table(gcp_project_id, cbt_instance_id, cbt_table_name, credentials)
    gcs_bucket = gcs_load_bucket(gcp_project_id, bucket_id, credentials)
    return cbt_table, gcs_bucket

def gcs_load_weights(model, bucket, prefix, tmp_weights_filepath):
    model_prefix = prefix + '_model'
    blobs_list = bucket.list_blobs(max_results=10, prefix=model_prefix)
    newest_blob = None
    for i,blob in enumerate(blobs_list):
        if i == 0: newest_blob = blob
        elif blob.time_created > newest_blob.time_created:
            newest_blob = blob
    if newest_blob is not None:
        # if model.public_url == newest_blob.public_url: return
        try: newest_blob.download_to_filename(tmp_weights_filepath)
        except:
            print("-> Model [{}] is currently being written.".format(model.public_url))
            return
        model.load_weights(tmp_weights_filepath)
        model.public_url = newest_blob.public_url
        print("-> Fetched most recent model [{}].".format(model.public_url))
    else:
        print("-> No models match the prefix \'{}\'.".format(model_prefix))

def gcs_save_weights(model, bucket, tmp_weights_filepath, model_filename):
    model.save_weights(tmp_weights_filepath)
    blob = bucket.blob(model_filename)
    blob.upload_from_filename(tmp_weights_filepath)
    print("-> Saved model to bucket as [{}].".format(model_filename))

def cbt_global_iterator(cbt_table):
    row_filter = row_filters.CellsColumnLimitFilter(1)
    gi_row = cbt_table.read_row('global_iterator'.encode())
    if gi_row is not None:
        global_i = gi_row.cells['global']['i'.encode()][0].value
        global_i = struct.unpack('i', global_i)[0]
    else:
        gi_row = cbt_table.row('global_iterator'.encode())
        gi_row.set_cell(column_family_id='global',
                        column='i'.encode(),
                        value=struct.pack('i',0),
                        timestamp=datetime.datetime.utcnow())
        cbt_table.mutate_rows([gi_row])
        global_i = 0  
    return global_i