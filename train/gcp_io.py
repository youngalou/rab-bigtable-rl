from google.oauth2 import service_account
from google.cloud import bigtable
from google.cloud import storage

def cbt_load_table(gcp_project_id, cbt_instance_id, cbt_table_name, credentials):
    print('-> Looking for the [{}] table.'.format(cbt_table_name))
    client = bigtable.Client(gcp_project_id, admin=True, credentials=credentials)
    instance = client.instance(cbt_instance_id)
    table = instance.table(cbt_table_name)
    if not table.exists():
        print("-> Table doesn't exist. Creating [{}] table...".format(cbt_table_name))
        max_versions_rule = bigtable.column_family.MaxVersionsGCRule(1)
        column_families = {'trajectory': max_versions_rule}
        table.create(column_families=column_families)
        print('-> Table created. Give it ~60 seconds to initialize before loading data.')
        exit()
    else:
        print("-> Table found.")
    return table

def gcs_load_weights(gcp_project_id, bucket_id, credentials, model_prefix, tmp_weights_filepath):
    print('-> Looking for the [{}] bucket.'.format(bucket_id))
    storage_client = storage.Client(gcp_project_id, credentials=credentials)
    bucket = storage_client.get_bucket(bucket_id)
    print('-> Bucket found.')
    blobs_list = bucket.list_blobs(max_results=10, prefix=model_prefix)
    newest_blob = None
    for i,blob in enumerate(blobs_list):
        if i == 0: newest_blob = blob
        elif blob.time_created > newest_blob.time_created:
            newest_blob = blob
    if newest_blob is not None:
        print("-> Fetched most recent model [{}].".format(newest_blob.public_url))
        newest_blob.download_to_filename(tmp_weights_filepath)
    else:
        print("-> No models match the prefix [{}].".format(model_prefix))
    return bucket, True if newest_blob is not None else False

def gcs_save_model(model, bucket, tmp_weights_filepath, model_filename):
    model.save_weights(tmp_weights_filepath)
    blob = bucket.blob(model_filename)
    blob.upload_from_filename(tmp_weights_filepath)
    print("-> Saved model to bucket as [{}].".format(model_filename))