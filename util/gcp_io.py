import struct
import datetime
import numpy as np

from google.oauth2 import service_account
from google.cloud import storage
from google.cloud import bigtable
from google.cloud.bigtable import row_filters

def gcs_load_bucket(gcp_project_id, bucket_id, credentials):
    """ returns a gcs_bucket object.

        gcp_project_id --  string (default none)
        bucket_id -- string (default none)
        credentials -- json file path (default none)
    """
    print('-> Looking for the [{}] bucket.'.format(bucket_id))
    storage_client = storage.Client(gcp_project_id, credentials=credentials)
    gcs_bucket = storage_client.get_bucket(bucket_id)
    print('-> Bucket found.')
    return gcs_bucket

def cbt_load_table(gcp_project_id, cbt_instance_id, cbt_table_name, credentials):
    """ returns a bigtable object.

        gcp_project_id --  string (default none)
        cbt_instance_id -- string (default none)
        cbt_table_name -- string (default none)
        credentials -- json file path (default none)
    """
    print('-> Looking for the [{}] table.'.format(cbt_table_name))
    client = bigtable.Client(gcp_project_id, admin=True, credentials=credentials)
    instance = client.instance(cbt_instance_id)
    cbt_table = instance.table(cbt_table_name)
    if not cbt_table.exists():
        print("-> Table doesn't exist. Creating [{}] table...".format(cbt_table_name))
        max_versions_rule = bigtable.column_family.MaxVersionsGCRule(1)
        column_families = {'step': max_versions_rule, 'global': max_versions_rule}
        cbt_table.create(column_families=column_families)
        print('-> Table created. Give it ~10 seconds to initialize before loading data.')
        exit()
    else:
        print("-> Table found.")
    return cbt_table

def gcp_load_pipeline(gcp_project_id, cbt_instance_id, cbt_table_name, bucket_id, credentials):
    """ returns a (bigtable object, gcs bucket object).

        gcp_project_id --  string (default none)
        cbt_instance_id -- string (default none)
        cbt_table_name -- string (default none)
        bucket_id -- string (default none)
        credentials -- json file path (default none)
    """
    cbt_table = cbt_load_table(gcp_project_id, cbt_instance_id, cbt_table_name, credentials)
    gcs_bucket = gcs_load_bucket(gcp_project_id, bucket_id, credentials)
    return cbt_table, gcs_bucket

def gcs_load_weights(model, bucket, prefix, tmp_weights_filepath):
    """ Downloads weights from bucket then loads weights to model.

        model -- tensorflow model class (default none)
        bucket -- gcs bucket object (default none)
        prefix -- string (default none)
        tmp_weights_filepath -- filepath (default none)
    """
    model_prefix = prefix + '_model'
    blobs_list = bucket.list_blobs(max_results=10, prefix=model_prefix)
    newest_blob = None
    for i,blob in enumerate(blobs_list):
        if i == 0: newest_blob = blob
        elif blob.time_created > newest_blob.time_created:
            newest_blob = blob
    if newest_blob is not None:
        # if model.public_url == newest_blob.public_url: return
        try:
            newest_blob.download_to_filename(tmp_weights_filepath)
            model.load_weights(tmp_weights_filepath)
        except:
            print("-> Model [{}] cannot be loaded.".format(newest_blob.public_url))
            return
        model.public_url = newest_blob.public_url
        print("-> Fetched most recent model [{}].".format(model.public_url))
    else:
        print("-> No models match the prefix \'{}\'.".format(model_prefix))

def gcs_save_weights(model, bucket, tmp_weights_filepath, model_filename):
    """ Saves weights from model then uploads weights to bucket

        model -- tensorflow model class (default none)
        bucket -- gcs bucket object (default none)
        tmp_weights_filepath -- filepath (default none)
        model_filename -- string (default none)
    """
    model.save_weights(tmp_weights_filepath)
    blob = bucket.blob(model_filename)
    blob.upload_from_filename(tmp_weights_filepath)
    print("-> Saved model to bucket as [{}].".format(model_filename))

def cbt_global_iterator(cbt_table):
    """ Fetches and sets global iterator from bigtable.

        cbt_table -- bigtable object (default none)
    """
    row_filter = row_filters.CellsColumnLimitFilter(1)
    gi_row = cbt_table.read_row('collection_global_iterator'.encode())
    if gi_row is not None:
        global_i = gi_row.cells['global']['i'.encode()][0].value
        global_i = struct.unpack('i', global_i)[0] + 1
    else:
        global_i = 0
    gi_row = cbt_table.row('collection_global_iterator'.encode())
    gi_row.set_cell(column_family_id='global',
                    column='i'.encode(),
                    value=struct.pack('i',global_i),
                    timestamp=datetime.datetime.utcnow())
    cbt_table.mutate_rows([gi_row])
    return global_i

def cbt_global_trajectory_buffer(cbt_table, local_traj_buff, global_traj_buff_size):
    row_filter = row_filters.CellsColumnLimitFilter(1)
    old_row = cbt_table.read_row('global_traj_buff'.encode())
    if old_row is not None:
        global_traj_buff = np.frombuffer(old_row.cells['global']['traj_buff'.encode()][0].value, dtype=np.int32)
        global_traj_buff = np.append(global_traj_buff, local_traj_buff)
        update_size = local_traj_buff.shape[0] - (global_traj_buff_size - global_traj_buff.shape[0])
        if update_size > 0: global_traj_buff = global_traj_buff[update_size:]
    else:
        global_traj_buff = local_traj_buff
    new_row = cbt_table.row('global_traj_buff'.encode())
    new_row.set_cell(column_family_id='global',
                 column='traj_buff'.encode(),
                 value=global_traj_buff.tobytes(),
                 timestamp=datetime.datetime.utcnow())
    cbt_table.mutate_rows([new_row])

def cbt_get_global_trajectory_buffer(cbt_table):
    row_filter = row_filters.CellsColumnLimitFilter(1)
    row = cbt_table.read_row('global_traj_buff'.encode())
    if row is not None:
        return np.flip(np.frombuffer(row.cells['global']['traj_buff'.encode()][0].value, dtype=np.int32), axis=0)
    else:
        print("Table is empty.")
        exit()

def cbt_read_rows(cbt_table, prefix, num_rows, global_i):
    """ Reads N(num_rows) number of rows from cbt_table, starting from the global iterator value.

        cbt_table -- bigtable object (default none)
        prefix -- string (default none)
        num_rows -- integer (default none)
        global_i -- integer (default none)

    """
    start_i, end_i = global_i - num_rows, global_i - 1
    start_row_key = prefix + '_trajectory_' + '{:05d}'.format(start_i)
    end_row_key = prefix + '_trajectory_' + '{:05d}'.format(end_i)
    partial_rows = cbt_table.read_rows(start_row_key, end_row_key, limit=num_rows, end_inclusive=True)
    return [row for row in partial_rows]

def cbt_read_row(cbt_table, prefix, row_i):
    """ Reads a row from cbt_table indexed by prefix + row_i.

        cbt_table -- bigtable object (default none)
        prefix -- string (default none)
        row_i -- integer (default none)
    """
    row_key = prefix + '_trajectory_' + str(row_i)
    row = cbt_table.read_row(row_key)
    return row

def cbt_read_trajectory(cbt_table, traj_i):
    """ Reads N(num_rows) number of rows from cbt_table, starting from the global iterator value.

        cbt_table -- bigtable object (default none)
        prefix -- string (default none)
        num_rows -- integer (default none)
        global_i -- integer (default none)

    """
    start_row_key = 'traj_{:05d}_step_{:05d}'.format(traj_i, 0)
    end_row_key = 'traj_{:05d}_step_{:05d}'.format(traj_i+1, 0)
    partial_rows = cbt_table.read_rows(start_row_key, end_row_key)
    return [row for row in partial_rows]