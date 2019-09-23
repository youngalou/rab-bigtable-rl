# Example docker build -t gcr.io/for-robolab-cbai/rab16:init .  --build-arg git_pull=true env_filename=CraneML
FROM gcr.io/for-robolab-cbai/env:1.0
#Use git_pull=True during build to pull from from latest git
ARG git_pull
ARG env_filename

ENV env_filepath=deploy/${env_filename}
ENV env_module=crane.collect
ENV env_env_filename=${env_filename}
WORKDIR /root/rab-bigtable-rl
RUN if [ "$git_pull" ] ; then git pull; else echo not pulling; fi
#Example gs bucket link https://storage.googleapis.com/rab_builds/CraneML_Linux.zip
RUN if [ "$env_filename" ] ; then \
    curl https://storage.googleapis.com/rab_builds/${env_filename}.zip -o deploy/${env_filename}.zip &&\
    unzip deploy/${env_filename}.zip -d deploy/ \
    ; else echo leaving default env \
    ; fi
CMD python3 -m \
    ${env_module} \
    --docker-training=True \
    --env-filename=deploy/${env_env_filename}.x86_64