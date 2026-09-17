# Frequently Asked Questions (i.e. troubleshooting)

## Setup and Workflow

### I can't run any of the shell scripts in the `scripts` folder. I keep getting a `Permission Denied` error.

You need to change the permissions on these files so that you can run them. From home, first change directories into the scripts folder by running `cd <your repo name>/course/scripts`.  Once you are in `scripts`, run the command `ls -la` (this will list all the files in this folder in long form (-l) and include hidden files in the list (-a)). The output will look something like this:

```
total 18
drwx------ 2 <your username> grp_202 4096 Sep 15 17:40 .
drwx------ 2 <your username> grp_202 4096 Sep 15 17:50 ..
-rw------- 1 <your username> grp_202 1755 Sep 15 17:40 check_gpu_env.py
-rw------- 1 <your username> grp_202 3029 Sep 15 17:40 check_login_env.py
-rw------- 1 <your username> grp_202  650 Sep 15 17:40 cluster.env
-rw------- 1 <your username> grp_202 1065 Sep 15 17:40 gpu_interactive.sh
-rw------- 1 <your username> grp_202 2348 Sep 15 17:40 kill_my_jobs.sh
-rw------- 1 <your username> grp_202 2313 Sep 15 17:40 my_jobs.sh
-rw------- 1 <your username> grp_202 2517 Sep 15 17:40 my_usage.sh
-rw------- 1 <your username> grp_202 2206 Sep 15 17:40 train.sbatch
```

The first character in each row is a `d` (meaning directory) or `-` (meaning a file). The next three characters are the permissions for the owner of the file, the next three characters are the permissions for the group that owns the file, and the last three characters are the permissions of all other users. If a character is `-`, that category of users doesn't have that permision. `r` means read permissions, `w` means write permissions, and `x` means execute permissions.

In the example above, you have `r` and `w` permissions for all these files, but not `x` permissions. That is why you are getting the `Permission Denied` error.

To change this, you use the comand `chmod +x <file 1> <file 2> ... <file n>`. Check that it works by running `ls -la` again. If you run `chmod +x my_jobs.sh my_usage.sh kill_my_jobs.sh` the permissions should now say:

```
total 18
drwx------ 2 mlusardi grp_202 4096 Sep 15 17:40 .
drwx------ 2 mlusardi grp_202 4096 Sep 15 17:50 ..
-rw------- 1 mlusardi grp_202 1755 Sep 15 17:40 check_gpu_env.py
-rw------- 1 mlusardi grp_202 3029 Sep 15 17:40 check_login_env.py
-rw------- 1 mlusardi grp_202  650 Sep 15 17:40 cluster.env
-rw------- 1 mlusardi grp_202 1065 Sep 15 17:40 gpu_interactive.sh
-rwx------ 1 mlusardi grp_202 2348 Sep 15 17:40 kill_my_jobs.sh
-rwx------ 1 mlusardi grp_202 2313 Sep 15 17:40 my_jobs.sh
-rwx------ 1 mlusardi grp_202 2517 Sep 15 17:40 my_usage.sh
-rw------- 1 mlusardi grp_202 2206 Sep 15 17:40 train.sbatch
```