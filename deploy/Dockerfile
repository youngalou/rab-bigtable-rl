# Example docker build -t gcr.io/for-robolab-cbai/rab16:1.4 .  --build-arg git_pull=true
FROM gcr.io/for-robolab-cbai/rab16:1.4
#Use git_pull=True during build to pull from from latest git
ARG git_pull
ARG env_filename

ENV env_game_exe=CraneML.x86_64
ENV env_module=crane.collect
WORKDIR /root/rab-bigtable-rl
RUN if [ "$git_pull" ] ; then git pull; else echo not pulling; fi
#RUN if [ "$env_filename" ] ; then 
CMD python3 -m \
    ${env_module} \
    --docker-training=True \
    --env-filename=${env_game_exe}