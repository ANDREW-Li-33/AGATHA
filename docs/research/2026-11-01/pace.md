# TOPIC pace: PACE ICE (Georgia Tech Instructional Cluster Environment) as compute for Stage 1, as of 2026-11-01

## PACE-1 [high CRIT] Q1 Eligibility: who can use ICE, and is it restricted to coursework?
CLAIM: ICE access comes through courses and workshops, not individual requests. Instructors apply, and students cannot apply for themselves.
EVID: The Getting Started article says ICE is available to GT faculty to teach classes or workshops, with the instructor applying using course-specific information, and it states 'Students should not fill out the ICE application.' Participation in PACE (KB0042194) labels ICE as being for academic classes and sends research users to Phoenix. Archived Fall 2021 ICE orientation slides say students get access automatically from registrar enrollment once the course is enabled (https://web.archive.org/web/2023/https://docs.pace.gatech.edu/ice_cluster/img/ICE_orientation_fall2021.pdf).
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102
IMPACT: First question for Andrew: do they have ICE access now, and through which Fall 2026 course? A reachable port 22 says nothing about whether their account exists.
VERDICT: confirmed

## PACE-2 [medium CRIT] Q1 Eligibility: is a personal or research project not tied to a course allowed?
CLAIM: Inference: a non-course project has no stated permission on ICE. I found no PACE wording that expressly allows or forbids it, so treat it as not cleared until an instructor or PACE confirms.
EVID: Every readable PACE page defines ICE by classes, workshops and training. GT's Acceptable Use Policy requires IT resources to be used per their 'intended use in support of the Institute's mission'. Its employee clause lists scholarly activity, research and learning as legitimate business, and its student personal-use clause covers only the ResNet, EastNet and LAWN networks. The ICE request form requires sign-in, so any terms there are unread.
SRC: https://policylibrary.gatech.edu/information-technology/acceptable-use-policy
IMPACT: Ask Andrew whether this project is part of a course or has an instructor's approval. Do not describe ICE use as approved until that is answered.
VERDICT: partially_correct
CORRECTED: Inference: a non-course project has no stated permission on ICE. No readable PACE wording expressly allows or forbids it, but PACE's request routes separate the two uses: research goes to Phoenix through the research form, and classes and events go to ICE. CoC's form describes student use as 'course assignments'. Treat it as not cleared until an instructor or TA confirms. A project tied to an enrolled course section, including VIP or special projects, is a course route.
VERIF-EVID: The conclusion holds, but two evidence lines are too strong. 'The ICE request form requires sign-in' is true only for the non-CoC ServiceNow form. CoC's Qualtrics form opens without sign-in, and its first page asks for VIP team codes. I did not go past page 1. 'Every readable PACE page defines ICE by classes, workshops and training' overreaches: PACE's citation page says 'If you used the ICE cluster or the AI Makerspace for your work, please acknowledge', though that is not a permission. CoC's guidelines cover 'class projects' and rank 'special projects course needs' after core and breadth courses. pace.gatech.edu/participation lists 'For Research' and 'For Academic Classes & Other Events' separately. The AUP quote is accurate, and its full sentence starts 'in accordance with applicable licenses and contracts'.
VERIF-SRC: https://gatech.co1.qualtrics.com/jfe/form/SV_3ylNMNNFa1B3tcO

## PACE-3 [high CRIT] Q1 Eligibility: who supports ICE students?
CLAIM: PACE does not support ICE students directly. Students go to their instructor or TA, who may then contact PACE.
EVID: The Getting Started article states this support structure. The archived 2021 orientation slides add that students may not open PACE support tickets directly.
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042102
IMPACT: Eligibility and quota questions should go through Andrew's instructor or TA, so we need to know who that is. Emailing pace-support directly may not get a student an answer.
VERDICT: confirmed

## PACE-4 [high CRIT] Q1 Eligibility: what is the documented route if ICE does not fit?
CLAIM: Phoenix is PACE's documented home for research use. It needs a faculty sponsor, or a PACESHIP award for students without advisors, and PACESHIP review cannot be counted on before Nov 5.
EVID: KB0042194 says GT faculty request Phoenix accounts for students. The base tier gives 10,000 CPU-hours a month, 20 GB home, 15 TB scratch and 1 TB project storage, and gpu-l40s is listed at $0.3444 per GPU-hour internal after Oct 1, 2026. PACESHIP accepts undergraduate or graduate students on a rolling basis, with review 'within 2 weeks' typically (https://pace.gatech.edu/evpr-paceship-application/). The CoC Fall 2025 grad orientation says CoC students have access to Phoenix (https://support.cc.gatech.edu/sites/default/files/2025-grad-orientation.pdf).
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042194
IMPACT: Ask whether Andrew has a faculty advisor and which college they are in. Inference: if ICE access is missing, one-segment Stage 1 is CPU-only and could run on Andrew's Mac as the Nov 5 fallback.
VERDICT: partially_correct
CORRECTED: Phoenix is PACE's documented home for research use. Access comes through a faculty PI, a PACESHIP award for students without advisors, or a school/college/unit allocation run by distributed IT, which PACE says is generally intended for groups including 'Students with no PI'. PACESHIP review cannot be counted on before Nov 5.
VERIF-EVID: Base-tier figures and the $0.3444 gpu-l40s rate match KB0042194. PACESHIP says 'Typically, applications will be reviewed within 2 weeks' and 'may take longer'. It also requires enrollment 'for classes at least for a full semester at the time of application'. pace.gatech.edu/participation names only undergraduates for PACESHIP, while the application page and KB0042194 cover students generally. The claim misses a third route in KB0043668: school allocations are 'generally intended for' categories that include 'Students with no PI'. The Fall 2026 CoC grad orientation, newer than the 2025 copy cited, repeats 'CoC students have access to ... PACE Phoenix'. For a CoC student, asking TSO may be faster than PACESHIP (inference). See 'missed' for a limit on the Mac fallback.
VERIF-SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0043668

## PACE-5 [high CRIT] Q2 Access: login host and VPN requirement
CLAIM: Log in by SSH to login-ice.pace.gatech.edu with a GT username. The client-based GlobalProtect VPN is required unless on a wired campus-building connection, and since August 2026 that includes eduroam and dorms.
EVID: The Aug 5, 2026 maintenance post lists the ICE firewall change, says wired campus-building connections keep direct access, and says the clientless browser VPN is not recommended. The Aug 13 release post repeats the eduroam change (https://pace.gatech.edu/2026/08/13/maintenance-ice-and-firebird-released/). The host and GT credentials come from the archived 2023 Log on to ICE page (https://web.archive.org/web/20230922025245/https://docs.pace.gatech.edu/ice_cluster/ice_logon/); the current article KB0042100 needs sign-in.
SRC: https://pace.gatech.edu/2026/08/05/maintenance-pace-upcoming-maintenance-aug-11th-to-13th/
IMPACT: GlobalProtect on Andrew's Mac is setup step 1 and a human step each session. Older notes saying eduroam alone is enough are out of date.
VERDICT: confirmed

## PACE-6 [medium] Q2 Access: login node pool
CLAIM: Inference from DNS: ICE has at least two login nodes behind a load balancer, so separate connections to login-ice can land on different machines.
EVID: DNS lookups on 2026-11-01, with no connection to any PACE host: login-ice is a CNAME to login-ice.f5dns.gatech.edu, which returned 128.61.254.151 and then .154 in two lookups. login-ice-1 maps to login-ice-gnr-1 (.151) and login-ice-2 maps to login-ice-gnr-2 (.154). The cited May 2026 post lists an upgrade of ICE login nodes to new hardware. A GT user's dotfiles pin login-ice-2 (https://github.com/sbryngelson/dotfiles/blob/master/.ssh/config).
SRC: https://pace.gatech.edu/2026/05/12/happening-now-pace-maintenance-may-12-15-2026/
IMPACT: Pin one host such as login-ice-1 in ~/.ssh/config so tmux sessions and the ControlMaster socket stay on one node.

## PACE-7 [low CRIT] Q2 Access: SSH key support and Duo/2FA
CLAIM: Unconfirmed: no readable PACE page says whether outside SSH keys work on ICE or whether SSH itself prompts Duo. Third-party notes conflict, and the most recent ones describe a password plus Duo push.
EVID: Official: the archived 2023 ICE page describes password login, and KB0042205 ties Duo to the GT VPN, with ssh-keygen steps only for node-to-node login. Third party on ICE: an Oct 2026 repo says password then Duo push (https://github.com/IncorvaiaM1/Options-Pricing-Through-Generative-Market-Simulation/blob/f9419afa12d8826170042bc26570236e576a553d/PACE.md); a 2025 repo uses ssh-copy-id (https://github.com/youssefg7/dl-sound-classification/blob/7ddbc19abda27d53672cb72a1f48235fde962d0c/PACE-ICE-SETUP.md); a repo last pushed Aug 2026 tests key login with BatchMode but also carries a stale 15 GB home figure (https://github.com/kspatel29/deep-learning-final-project/blob/6c7ba358ffd1fcc06b2322dfdcb6133e108410a2/PACE_USAGE.md). Third party on Phoenix, Sept 2026: 'SSH keys don't skip Duo on PACE' (https://github.com/abbybonde/BCARS-Common-Language/blob/81d68d6e9b988a39dc789e7b00183c7e016ce484/pace_pipeline/PACE_HOWTO.md).
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042205
IMPACT: Plan for an interactive login that only Andrew can complete; Claude must not handle the password. Ask Andrew what they see at login, or have them run the probe in exact_reference.
VERDICT: partially_correct
CORRECTED: Unconfirmed: no readable PACE page says whether outside SSH keys work on ICE or whether SSH itself prompts Duo. Third-party ICE notes suggest two paths rather than a true conflict: password login followed by a Duo push, and key login with no Duo. Plan for Andrew to complete the first login, then test key login.
VERIF-EVID: The gap in PACE documentation holds: none of the 616 blog posts mentions Duo or 2FA, and guest KB search finds only KB0042205, which ties Duo to the VPN. 'The most recent ones describe a password plus Duo push' is incomplete. Password path: Incorvaia, Oct 27, 2026, and a Deja-View plan, Oct 2026. Key path: kharon ENV.md, 'confirmed 2026-05-29', says 'Access via SSH key only (no Duo for key auth)'; kspatel, Aug 4, 2026, tests BatchMode; and the same HAAG lab cited in PACE-8 wrote on Oct 11, 2026 that key auth 'logs in without a password'. The only note saying keys do not skip Duo is about Phoenix. All of this is third-party and unverified. KB0042100 and KB0041969 need sign-in and may settle it.
VERIF-SRC: https://github.com/UjaanRakshit/kharon/blob/main/ENV.md

## PACE-8 [medium CRIT] Q2 Access: is SSH connection multiplexing permitted?
CLAIM: No PACE statement permits or forbids SSH multiplexing, but it is in active third-party use, including on ICE. Inference: it is tolerated, and it is probably the practical way to run repeated commands after one authentication.
EVID: PACE Policies and the head-node limits article do not mention it. A GT lab's ICE wrapper script enables ControlMaster with a 10-minute persist. A Phoenix how-to recommends the same setup so Duo is approved once (same BCARS URL as PACE-7). ContractLens notes suggest a plain interactive session first if the Duo step does not render (https://github.com/NB670/ContractLens/blob/d59c83e72ffd241e9b857641be3a7c51a0bc598b/docs/PACE_FINETUNE.md).
SRC: https://github.com/Human-Augment-Analytics/pfas-environment-cleanup/blob/dc96f2151ddabd19e442dc355fb61a0df374f08c/scripts/dft_wrapper.py
IMPACT: Use ControlMaster with a pinned host: Andrew opens the master connection, then commands reuse it. Keep commands sent over it light, because login-node limits still apply.
VERDICT: partially_correct
CORRECTED: No PACE statement permits or forbids SSH multiplexing. Third-party evidence on ICE conflicts: a note dated May 29, 2026 says the server blocks it, and the cited ICE script does not prove reuse works. Phoenix notes do recommend it. Treat multiplexing on ICE as untested, test a second session once, and keep key login as the fallback.
VERIF-EVID: PACE Policies and KB0041931 are silent, and no PACE post mentions it. The inference that it is tolerated is undermined by kharon ENV.md: 'SSH multiplexing is blocked by the server.' That note postdates the May 2026 ICE login-node hardware upgrade. The cited dft_wrapper.py does enable ControlMaster with a 10-minute persist, but the lab's own doc requires key login. OpenSSH mux.c returns -1 after a failed mux session request, which appears to let ssh continue on a normal connection, so the script would still appear to work with keys (my reading of the code). A Deja-View ICE plan used a ControlMaster socket, but the ICE lane was removed on Oct 19, 2026, and I found no sign it ran. The BCARS note is Phoenix, and ContractLens is advice for a Duo problem it had not confirmed on ICE. All third-party, so this does not refute the claim.
VERIF-SRC: https://github.com/UjaanRakshit/kharon/blob/main/ENV.md

## PACE-9 [high CRIT] Q2 Access: rules for login nodes
CLAIM: ICE login nodes enforce per-user hard limits of one CPU-equivalent, 4 GB of total memory and 5000 open files. Exceeding 4 GB kills processes with SIGKILL (exit 137) and no error message.
EVID: KB0041931 sets these limits for Phoenix and ICE from May 26, 2023, and encourages compute nodes for pip and conda installs and compiling. Compute nodes allow 131072 open files. PACE Policies (KB0042198) also bars processes using 90% or more of a CPU for over 30 minutes or 20% of memory, and PACE may terminate them without notice. Spack and Apptainer are unavailable on login nodes (KB0044244, KB0043878).
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0041931
IMPACT: Run all Parquet loading, projection, env creation and installs in jobs. Inference: a camera_image Parquet read on a login node could die silently and look like a code bug.
VERDICT: confirmed

## PACE-10 [high CRIT] Q3 Storage: quotas, backups and the quota command
CLAIM: ICE gives each user 30 GB of home with daily snapshots and 300 GB of scratch capped at 1 million files. Scratch is not backed up, and pace-quota reports usage.
EVID: Storage on ICE: home is on NetApp and scratch on Lustre over InfiniBand. Jobs also get local disk through ${TMPDIR}, cleared at job end. Extra quota is requested by instructors and TAs. One third-party repo last pushed Aug 2026 still says 15 GB of home, the figure on the 2023 page, so pace-quota is the authority for Andrew's account.
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042094
IMPACT: Keep code in home and in git, and put data and envs in ~/scratch. Size the Waymo download against 300 GB, and keep anything hard to recreate off scratch.
VERDICT: confirmed

## PACE-11 [high CRIT] Q3 Storage: retention and purge rules, including at semester end
CLAIM: At the end of each semester ICE deletes scratch files not touched in 120 days. Home files are deleted after a user has had no ICE access or login for a year. PACE's 45-day monthly scratch policy does not list ICE in its scope.
EVID: KB0042094 gives the ICE rules. KB0042199 scopes the 45-day policy to Phoenix and Hive. An April 2024 post says the ICE spring deletion ran the week of May 13 (https://pace.gatech.edu/2024/04/22/pace-maintenance-period-may-07-may-10-2024/). Inference from arithmetic: a file written Nov 2, 2026 reaches 120 days on Mar 2, 2027, so the first sweep that could hit it is at the end of Spring 2027. PACE does not define 'touched' for this rule.
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042094
IMPACT: No purge risk to Stage 1 this term. Keep lock files and manifests in git so scratch contents can be rebuilt.
VERDICT: confirmed

## PACE-12 [low CRIT] Q3 Storage: default permissions and whether other users can read files
CLAIM: Default permissions on ICE home and scratch are not documented in anything I could read, so assume other users might be able to read files until checked.
EVID: Storage on ICE discusses permissions only for course shared directories. PACE Policies says users have no expectation of privacy from authorized personnel. Third-party repos show home at /home/hice1/<user> and ~/scratch linking to /storage/ice1/<x>/<y>/<user>, but none state modes. The Storage Guide (KB0041972) needs sign-in.
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042094
IMPACT: Before any Waymo data lands, check modes with ls -ld and restrict the data directory to mode 700. The design doc says to keep access restricted until licensing is cleared.
VERDICT: confirmed

## PACE-13 [high CRIT] Q4 Slurm: walltime and per-user limits
CLAIM: Each ICE job can use up to 512 CPU-hours or 16 GPU-hours, with walltime up to 18 h for CPU and 16 h for GPU. A job with no resource request gets 1 core, 1 GB per core and 1 hour. Each user may have 50 GPU jobs queued and 32 GPU-hours running.
EVID: Using Slurm on ICE (KB0042096). The GPU limits took effect Sept 22, 2026, down from 500 jobs, after reports of long GPU queues (https://pace.gatech.edu/2026/09/22/ice-scheduler-improvements/). I found no published per-user CPU limit.
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096
IMPACT: Stage 1 fits easily: 24 cores for 18 h is 432 CPU-hours, while 64 cores caps near 8 h. Stage 4 GPU work must checkpoint within 16 GPU-hours per job.
VERDICT: partially_correct
CORRECTED: Each ICE job is capped at 512 CPU-hours and 16 GPU-hours, and both caps apply to the same job. Walltime is up to 18 h for CPU and 16 h for GPU unless those caps cut it shorter. A job with no resource request gets 1 core, 1 GB per core and 1 hour. Each user may have 50 GPU jobs in the queue counting running and pending together, with further submissions rejected, and 32 GPU-hours of running GPU work.
VERIF-EVID: Minor wording fixes only; the numbers and the Stage 1 arithmetic are right. KB0042096 says 'a maximum of 512 CPU hours and 16 GPU hours', not 'or'. The Sept 22, 2026 post defines the 50-job limit as 'running and pending combined' and says it is 'Effective immediately', down from 500. KB0042096 adds 'Additional submissions will be rejected'. I found no published per-user CPU limit either.
VERIF-SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042096

## PACE-14 [high CRIT] Q4 Slurm: partitions, QOS and account flag
CLAIM: Do not pass -p on ICE, and no -A is needed: partitions and QOS are assigned automatically. Partitions are ice-cpu and ice-gpu for everyone, plus coc-cpu, coc-gpu, pace-cpu, pace-gpu and coe-gpu for course priority.
EVID: ICE Cluster Resources says not to request a partition. KB0042096 says assignment is automatic, and its ICE sbatch examples have no account flag and a blank Account line in the prolog. Optional QOS flags are -q pace-ice, coe-ice and coc-ice for students in several courses; coc-grade, coe-grade and pace-grade are for instructors and TAs. Inference: the -A flags in the venv, Apptainer and Spack guides are Phoenix examples.
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042095
IMPACT: Omit -p and -A from sbatch and salloc headers, and do not copy charge-account flags from general PACE guides.
VERDICT: confirmed

## PACE-15 [high] Q4 Slurm: does ICE have Open OnDemand?
CLAIM: Yes. ICE Open OnDemand is at https://ondemand-ice.pace.gatech.edu/ and offers Jupyter, interactive desktop, interactive shell and VS Code.
EVID: KB0042102 links it. KB0042133 requires the GT VPN, warns that Safari works poorly, and notes interactive jobs keep running after the browser closes. KB0044712 recommends the OnDemand VS Code app over desktop Remote Explorer, with at least 4 cores and 16 GB.
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042133
IMPACT: Good for inspecting box overlays in Jupyter. Inference: it also needs no local SSH setup, so it is a fallback if Duo makes SSH awkward. Andrew should use Chrome or Firefox, not Safari.

## PACE-16 [high] Q5 Hardware: CPU node specs
CLAIM: Most ICE CPU nodes have dual Xeon Gold 6226 (24 cores) and 192 GB of RAM. Larger options include 768 GB nodes, 64-core EPYC 7513 nodes with 512 GB, and one 192-core node with 1.5 TiB.
EVID: ICE Cluster Resources lists 39 CPU nodes at 192 GB with 1.6 to 1.9 TB NVMe, 1 at 384 GB with SAS, 7 at 768 GB, 4 EPYC 7513 nodes and 1 Xeon 6972P node. The scheduler reserves 8 GB per node, and every node has HDR100 InfiniBand.
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042095
IMPACT: A partial 24-core node is ample for Stage 1. Request memory explicitly, since the default is 1 GB per core.

## PACE-17 [high CRIT] Q5 Hardware: GPU models and whether an L40S exists on ICE
CLAIM: An NVIDIA L40S exists on ICE: 4 nodes, each with 8 L40S 48 GB cards (32 total), dual Xeon Gold 6548Y+ (64 cores), 768 GB of RAM and 3 x 3.84 TB NVMe.
EVID: The same table lists V100 16 GB, RTX 6000 24 GB, A100 40 and 80 GB, A40 48 GB, MI210 64 GB, H100 80 GB, H200 142 GB and RTX Pro 6000 Blackwell 48 GB. It reserves 13 H100 and 12 H200 nodes for CoE and AI Makerspace users. Request an L40S with --gres=gpu:L40S:1 (KB0042096). The October 2026 maintenance set models other than V100 and RTX6000 to driver 615-open with CUDA 13.4.2 (https://pace.gatech.edu/2026/10/05/maintenance-pace-october-5-8-maintenance-starting-now/).
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042095
IMPACT: The design doc's L40S is available, so no separate GPU question is needed now. Stage 1 needs no GPU. The 16 GPU-hour job limit and EGL or Vulkan support remain Stage 4 risks.
VERDICT: confirmed

## PACE-18 [medium CRIT] Q6 Network: outbound internet from compute nodes
CLAIM: Inference: ICE compute nodes have outbound internet. No PACE page says so directly, and reachability of Google Cloud Storage is untested.
EVID: PACE guides run network fetches on compute nodes: pip install in an interactive job (KB0044520), DockerHub and NGC pulls that must run on compute nodes (KB0043878), and Spack fetching from ghcr.io (KB0044244). ICE is described as using the same hardware and software as Phoenix, and all compute nodes have 10-gigabit Ethernet (KB0042194). Third party on ICE: a July 2026 note downloads COCO from a compute node (https://github.com/abhi25072002/DL-VLM-Hallucination/blob/0d2f47d2b3c357d03eb4b6025aa74764c50f10dc/PACE_SETUP.md), while an Oct 2026 note hedges that compute nodes may not have internet (PACE.md, same URL as PACE-7).
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0043878
IMPACT: Design for gs:// access from jobs, but run the curl test in exact_reference before relying on it.
VERDICT: partially_correct
CORRECTED: ICE compute nodes very likely have outbound internet. PACE's ICE Makerspace tutorial starts labs on compute nodes and teaches downloading datasets from HuggingFace Hub, which comes close to saying so directly. No page states it as a policy, and reachability of Google Cloud Storage is untested, so the curl test is still needed.
VERIF-EVID: KB0044522 is ICE-specific: 'This tutorial can be run on the instructional cluster (ICE)', the form is used 'to start the labs on compute nodes', and it shows 'how they can download a dataset from HuggingFace Hub'. KB0042188 has users open Firefox to authenticate from an Interactive Desktop on a compute node. KB0044712 gives an Apptainer recipe that runs apt and pip, and KB0043878 limits builds to compute nodes. Correction on the COCO note: PACE_SETUP.md was authored Mar 20, 2026, and only the repo push is July 2026. It also uses '-q inferno', a Phoenix QOS, so it is weak evidence. The Oct 2026 note's 'may not have internet' is a hedge, not a test. KB0044522 and KB0042188 are over a year old, and May 2026 included PACE firewall upgrades, so a live test still matters.
VERIF-SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0044522

## PACE-19 [high CRIT] Q6 Network: are large downloads on login nodes discouraged?
CLAIM: Yes. Large cloud transfers on login nodes are discouraged, so gs:// downloads should run inside a compute job.
EVID: A Dec 2025 PACE post asks users to avoid large cloud transfers through rclone or similar tools on the login node because of campus network load. PACE Policies says large data operations on head nodes should be limited. The older KB0041931 (2023) says head nodes can be used for scp, sftp and rclone. Inference: the newer post governs large cloud transfers.
SRC: https://pace.gatech.edu/2025/12/11/globus-connectors-for-cloud-storage-on-pace/
IMPACT: Put Waymo downloads in an sbatch script limited to the components for one segment. Small scp transfers to the Mac from a login node are acceptable.
VERDICT: partially_correct
CORRECTED: Yes. PACE asks users to avoid large cloud transfers through rclone or similar tools on the login node and names Globus connectors as the best route. It does not list Google Cloud Storage among them or say compute-node downloads are exempt. Running a one-segment gs:// download in a compute job is a sound inference, not a PACE rule.
VERIF-EVID: The Dec 11, 2025 post calls Globus Connectors 'the best way' for 'Dropbox, Box, and OneDrive' and asks users to avoid large transfers on the login node because they 'can cause heavy load on the campus network'. A compute node may not remove that network load. KB0042198 says large data operations 'should be limited'. KB0041931 still allows rclone on head nodes. Missing support for the compute-job step: KB0042188 calls rclone the next-best option when Globus is not available and runs its setup in an Interactive Desktop on a compute node. KB0042094 supports small SCP transfers.
VERIF-SRC: https://pace.gatech.edu/2025/12/11/globus-connectors-for-cloud-storage-on-pace/

## PACE-20 [medium] Q6 Network: data transfer nodes and Globus
CLAIM: ICE has Globus, and PACE recommends it for large transfers. A separate Science DMZ endpoint, PACE Grid, exists on request. No source lists a Google Cloud Storage connector.
EVID: KB0042094 recommends Globus for ICE. A June 2026 post describes Globus ICE returning during an outage (https://pace.gatech.edu/2026/06/16/pace-downtime-updates-june-16-2026/). The Dec 2025 post names connectors for Dropbox, Box and OneDrive and invites requests for others. KB0042207 describes PACE Grid. The ICE collection name is in KB0041890, which needs sign-in.
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042094
IMPACT: Inference: Globus suits moving results off ICE but does not replace gs:// reads. It has stayed reachable during some outages, so it can be a way to retrieve data then.

## PACE-21 [medium CRIT] Q7 Software: module system contents
CLAIM: Documented modules include python (python/3.10.10 in the example), anaconda3, uv/0.9.17, gcc, cuda and spack. Apptainer is a compute-node command. ffmpeg and google-cloud-sdk modules are unconfirmed.
EVID: Sources are KB0044520, KB0042096, KB0045066, KB0044244 and KB0043878. The archived Dec 2023 list (https://web.archive.org/web/20231206103105/https://docs.pace.gatech.edu/slurm-software/software_list/) shows anaconda3, git/2.35.2, opencv/4.5.4, python/3.9.12 and pytorch, with no ffmpeg or Google Cloud entry, but it predates RHEL9. The current lists KB0043494 and KB0041975 need sign-in. KB0043621 says new central software should usually benefit at least 5 users, and evaluation may take days to weeks.
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0045066
IMPACT: Run module spider first. Plan user-space installs for a GCS client or gcloud, and for ffmpeg, rather than waiting on a PACE install before Nov 5.
VERDICT: confirmed

## PACE-22 [high CRIT] Q7 Software: recommended way to manage Python environments
CLAIM: Use uv or venv rather than conda. PACE's uv guide applies to ICE, requires UV_CACHE_DIR under ~/scratch before creating environments, and advises against loading a python module alongside uv.
EVID: KB0045066 gives ~/scratch as the ICE equivalent of project space, and recommends building environments in a dedicated Slurm job and only activating them in compute jobs. KB0044520 says venvs are generally recommended over conda. KB0042204 advises against conda init. GT also runs a pull-through PyPI mirror at pypi.oit.gatech.edu (KB0045732).
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0045066
IMPACT: Set cache variables first, build the env in a Slurm job on scratch, and commit the requirements file to git.
VERDICT: confirmed

## PACE-23 [high CRIT] Q7 Software: home-quota traps from pip, uv and conda caches
CLAIM: Default tool locations sit in home and can fill it: ~/.cache/uv, ~/.local/share/uv/python, ~/.cache/pip, ~/.conda, ~/.apptainer/cache and ~/spack. Move them to ~/scratch before first use.
EVID: PACE warns about ~/.cache/uv (KB0045066), ~/spack (KB0044244) and multi-GB .sif images (KB0043878). The archived 2023 conda guide says environments default to ~/.conda and tells ICE users to skip the project-space symlink (https://web.archive.org/web/20230922020910/https://docs.pace.gatech.edu/slurm-software/conda/). The uv storage docs give ~/.local/share/uv/python; pip caches in ~/.cache/pip (https://pip.pypa.io/en/stable/topics/caching/); Apptainer caches in $HOME/.apptainer/cache (https://apptainer.org/docs/user/latest/build_env.html). Inference: uv hardlinks need the cache and env on one filesystem, so keep both on scratch.
SRC: https://docs.astral.sh/uv/reference/storage/
IMPACT: Add the ~/.bashrc exports in exact_reference before any install. Keep them to plain exports with no conda init, which PACE links to OnDemand session failures.
VERDICT: confirmed

## PACE-24 [high] Q7 Software: Apptainer
CLAIM: At PACE, Apptainer pulls, builds and runs happen only on compute nodes.
EVID: KB0043878 says to pull or build inside salloc or sbatch with --mem=16G or more, use --nv for GPUs, and use --no-home to avoid ~/.local clashes. It recommends project space over home for images, and ICE has no project space. A January 2025 maintenance list enabled self-service container builds on ICE (https://pace.gatech.edu/2025/01/06/pace-maintenance-period-january-13th-16th-2025/).
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0043878
IMPACT: Not needed for Stage 1, but it is an option for ffmpeg or awkward dependencies. Store .sif files in ~/scratch.

## PACE-25 [high CRIT] Q8 Maintenance: any window in November 2026?
CLAIM: No PACE maintenance window is scheduled before Nov 5, 2026 or elsewhere in November 2026. The next windows are tentatively Jan 11-13, 2027 and May 11-13, 2027.
EVID: The Oct 7, 2026 completion post gives those dates. GT's status API (https://api.status.io/1.0/status/5be9af0e5638b904c2030699), checked at 2026-11-01 20:35 UTC, returned empty active and upcoming maintenance, with ICE Cluster, Globus Endpoints and VPN Operational. The newest PACE post is dated Oct 22, and none of the 600 archived posts mentions November 2026. The PACE homepage still shows the past October window, so it is stale.
SRC: https://pace.gatech.edu/2026/10/07/maintenance-complete-phoenix-ice-firebird-released/
IMPACT: No scheduled blocker for Nov 5. Recheck status.gatech.edu on Nov 3 and Nov 4, because emergency work is announced with little notice.
VERDICT: confirmed

## PACE-26 [high] Q8 Maintenance: historical pattern
CLAIM: PACE policy sets four maintenance windows a year. Fall windows moved from November to October after 2022, and 2026 windows ran in January, May, August and October.
EVID: PACE Policies says windows occur 'four times annually'. Post titles show November windows in 2019, 2021 and 2022 (Nov 2-4), and October in 2020, 2023 (Oct 24-26, extended to Oct 30), 2025 (Oct 6-8) and 2026 (Oct 5-8). No fall 2024 window appears in the titles, and 2025 shows January, a March 17-21 planned downtime, May and October. Recent windows start at 6:00 AM and span 3 to 4 days. ICE returned early in August and October 2026, and the August window deleted all pending ICE jobs.
SRC: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042198
IMPACT: The pattern supports the January date. Inference: expect a new outage around Jan 11-13, and resubmit pending ICE jobs afterward if the August cleanup recurs.

## PACE-27 [high CRIT] Q8 Maintenance: outage risk before Nov 5
CLAIM: Unplanned outages are the real risk before Nov 5. ICE had a scheduler failure on Oct 29 and the GT VPN had trouble on Oct 31, both resolved, and 2026 has seen several short-notice shutdowns.
EVID: Oct 29: ICE Slurm stopped around midnight, restarted at 9:15 AM and was resolved at 10:58 AM EDT (https://status.io/pages/incident/5be9af0e5638b904c2030699/6ae34431fe3e5605cb85098c). Oct 31: a Rich Building power issue at 7:30 AM affected VPN access, resolved by 12:31 PM (https://status.io/pages/incident/5be9af0e5638b904c2030699/6ae5d13f3e42aa05ed5ae3ed). Apr 30: an emergency security shutdown interrupted running jobs (https://pace.gatech.edu/2026/04/30/happening-now-pace-emergency-security-maintenance/). May 26: a Coda cooling failure cancelled all jobs while login and storage stayed up (https://pace.gatech.edu/2026/05/26/urgent-coda-data-center-cooling-issue-impacting-pace-services/). Jun 16-17: a security outage came with 5 days' notice (https://pace.gatech.edu/2026/06/11/unplanned-pace-downtime-next-week-june-16-17/). PACE says cooling and rolling-patch work should reduce interruptions (https://pace.gatech.edu/2026/09/08/upcoming-improvements-to-research-computing-updated-pace-compute-rates/).
SRC: https://status.gatech.edu/pages/5be9af0e5638b904c2030699/rss
IMPACT: Do not depend on ICE being up during the Nov 5 call. Finish early, keep jobs restartable, and keep a local copy of results Andrew needs, subject to the Waymo license review.
VERDICT: confirmed

## PACE-28 [medium CRIT] Q1 and Q8: account lifecycle
CLAIM: An older ICE account may no longer work. The August 2026 maintenance cleaned out unused ICE accounts, and no readable page says when access ends after a course.
EVID: The Aug 5, 2026 list includes the item 'Clean out unused accounts'. A Dec 2025 post says ICE opened to Spring 2026 courses after the January window (https://pace.gatech.edu/2025/12/17/pace-maintenance-period-01-12-26-to-01-16-26/). One third-party July 2026 note reports ICE closing the connection right after the password with no Duo prompt, cause unknown (ContractLens URL in PACE-8).
SRC: https://pace.gatech.edu/2026/08/05/maintenance-pace-upcoming-maintenance-aug-11th-to-13th/
IMPACT: Have Andrew log in and run pace-quota today instead of assuming access. If the connection closes right after the password, inference: suspect missing ICE access. Also ask whether access continues into Spring 2027.
VERDICT: partially_correct
CORRECTED: An older ICE account may no longer work. The August 2026 maintenance plan listed cleaning out unused ICE accounts, but PACE gave no definition and no completion notice. CoC's ICE access is term-scoped: its form is titled 'Semesterly Access' and says 'Access is only granted for the upcoming term(s)'. No readable page gives the exact date access ends after a course.
VERIF-EVID: The Aug 5 post lists '[ICE cluster] Clean out unused accounts' as planned work; the Aug 13 release posts do not mention it, so 'cleaned out' is unverified. 'No readable page says when access ends' misses the CoC form wording. KB0042094 also ties home deletion to a user who 'has not had access to ICE', which shows access can lapse. The Dec 2025 post does say ICE opened to Spring 2026 courses on Jan 16. ContractLens does report ICE closing the connection after the password, but its 'no ICE access' link is inference, and that note blames a client limitation.
VERIF-SRC: https://gatech.co1.qualtrics.com/jfe/form/SV_3ylNMNNFa1B3tcO

# VERIFIER MISSED
- If Andrew has no ICE course this term, ICE probably cannot be added in time. CoC's guidelines say 'only faculty requests sent to academicresources@cc.gatech.edu prior to the start of the semester will be considered', and special-projects courses come after core and breadth courses. The form says access covers 'the upcoming term(s)'. ICE opened to Spring 2026 courses on Jan 16, 2026, so mid-January is the likely next entry point (inference). The non-CoC form is unread because it needs sign-in. https://support.cc.gatech.edu/facilities/instructional-labs/academic-hpc-resources
- CoC students have a GPU cluster they can request themselves. The Fall 2026 CoC grad orientation lists HCE@CoC (Futurama) as 'Requestable by any CS/CSE/CM/CX enrolled student for academic use', with L40S among its GPUs. The form and docs sit behind GT login, and whether a non-course project counts as 'academic use' is unstated. Ask Andrew's college and major. https://support.cc.gatech.edu/sites/default/files/2026-grad-orientation.pdf
- Phoenix has a third route besides a faculty PI and PACESHIP: school/college/unit allocations 'generally intended for' groups including 'Students with no PI' (KB0043668). For a CoC student without an advisor, asking TSO may be faster than the two-week PACESHIP review (inference). https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0043668
- VIP sections are a real ICE course route: CoC's form asks for VIP team codes, and PACE's ICE page quotes an Art & AI VIP class. If this project sits under an enrolled VIP section, that may answer the eligibility question, subject to the pre-semester deadline. https://pace.gatech.edu/ice-cluster/
- The Mac fallback in PACE-4 has a catch. The waymo-open-dataset-tf-2-12-0 and tf-2-11-0 wheels on PyPI are manylinux x86_64 only, so the official library will not install natively on a Mac. A Mac run would need direct Parquet reads with PyArrow, or a Linux VM or container. Check this before naming the Mac as the Nov 5 fallback. https://pypi.org/project/waymo-open-dataset-tf-2-12-0/
- The AUP's opening sentence says IT Resources must be used 'in accordance with applicable licenses and contracts'. That makes Waymo license compliance on ICE a GT policy matter too, not only a question for Claude and Supabase uploads. https://policylibrary.gatech.edu/information-technology/acceptable-use-policy
- The VPN failed more than once in October: a web-portal DNS issue on Oct 7, two brief interruptions on Oct 15 and the Oct 31 incident. From eduroam or dorms the VPN is a hard dependency, so a wired campus-building connection is the practical fallback. Globus stayed open during parts of the 2026 outages, or came back early, so it is the way to reach data when login is down (Apr 30, May 26 and June 11 posts). https://status.gatech.edu/pages/5be9af0e5638b904c2030699/rss
- rclone may be a simpler Google Cloud Storage client than gcloud. PACE documents installing it as a user-space binary (KB0042188), and rclone has a Tier 1 GCS backend with headless and service-account options. Whether it works with Waymo's bucket is untested. https://rclone.org/googlecloudstorage/
- For Stage 4, PACE documents EGL rendering on an L40S: KB0044609 runs 'vglrun -d egl' and shows 'OpenGL Renderer: NVIDIA L40S'. That narrows the EGL risk in PACE-17 for Interactive Desktops. Headless Vulkan and CARLA remain untested, and PACE warns that GPU rendering slows GPU compute. https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0044609
- AI Makerspace is an open lead for engineering students. KB0042095 lists 'AI Makerspace users' separately from CoE courses, and a May 2025 ECE article says rules and onboarding were coming. I found no published non-course access process. https://ece.gatech.edu/news/2025/05/ai-makerspace-turns-one-students-charting-course
- Several pages that could settle open questions redirected to GT sign-in, so I did not read them: KB0042100 (Log on to ICE), KB0041969 (Log on to Phoenix), KB0043573 (RHEL9 for ICE), KB0043494 (software stack), KB0041972 (Storage Guide), KB0041890 (Globus), KB0043842 and KB0044110 (Makerspace datasets), plus PACE's October 2026 orientation slides on github.gatech.edu. Andrew can open them with a GT login, which is the fastest way to resolve PACE-7, PACE-12 and PACE-21.

# UNRESOLVED
- Is a personal or non-course research project allowed on ICE? -> Andrew asks their instructor or TA, who can ask pace-support@oit.gatech.edu. The ICE request form needs GT sign-in and may contain terms.
- Does Andrew have a working ICE account, through which course, and does it continue into Spring 2027? -> Andrew logs in over VPN and runs pace-quota. The course and instructor can only come from Andrew. The end-of-access date should be confirmed with the instructor or TA.
- Do SSH keys work on ICE, and does SSH prompt Duo, with or without a key? -> Andrew runs the key probe in exact_reference, section 2. Read Log on to ICE (KB0042100) after signing in.
- Does PACE permit SSH ControlMaster, and is there a session length limit? -> No readable policy covers it. Ask through the instructor or TA if a definite answer is needed.
- Can ICE compute nodes reach storage.googleapis.com? -> Run the curl test in exact_reference, section 8, inside a short salloc job.
- What are the default modes on home and scratch, and can other ICE users read them? -> Run the ls and umask commands in exact_reference, section 3, on login. Apply mode 700 to the data directory regardless.
- Are ffmpeg, a Google Cloud SDK, and rclone available as modules on the current RHEL9 stack? -> Run module spider on ICE. Read RHEL9 Environment for ICE (KB0043573) and PACE Software Stack (KB0043494) after signing in.
- What are the per-user CPU limits, and are there per-QOS limits beyond those published? -> Run the read-only sacctmgr and scontrol commands in exact_reference, section 9. PACE publishes only per-job CPU limits and per-user GPU limits.
- When exactly does the Fall 2026 ICE scratch deletion run, and does 'touched' mean access time or modification time? -> Ask through the instructor or TA. It does not affect files created this month.
- What is the Globus collection name for ICE, and can Globus reach Google Cloud Storage for this dataset? -> Read Using Globus to Transfer Files (KB0041890) after signing in. PACE invites requests for additional connectors.
- 16 PACE KB articles and the ICE request form redirected to GT SSO and could not be read as a guest; the orientation PDF on github.gatech.edu also needs sign-in. I did not sign in. Findings on these topics rely on archived 2023 pages or third-party repos. -> Andrew can read them signed in. KB numbers: KB0042100, KB0043573, KB0043494, KB0041975, KB0041890, KB0041972, KB0041982, KB0041621, KB0041536, KB0042189, KB0041969, KB0042172, KB0042197, KB0042499, KB0043842, KB0042280.
- Could an unplanned outage or new emergency maintenance land before Nov 5? -> This cannot be ruled out. Check https://status.gatech.edu/ and https://pace.gatech.edu/pace-announcements-blog/ on Nov 3 and Nov 4, and read PACE emails.

# EXACT REFERENCE
NOTHING BELOW WAS RUN. PACE was not contacted (DNS lookups only). Replace <gtuser>.
Tags: [PACE] from PACE docs. [ADAPTED] PACE example with values changed. [3P] third-party pattern, not documented by PACE. [UPSTREAM] tool vendor docs.

=== 1. HOSTS AND URLS
SSH pool: login-ice.pace.gatech.edu (CNAME login-ice.f5dns.gatech.edu)
Pinned nodes (DNS 2026-11-01): login-ice-1.pace.gatech.edu -> login-ice-gnr-1 (128.61.254.151); login-ice-2.pace.gatech.edu -> login-ice-gnr-2 (128.61.254.154)
OnDemand: https://ondemand-ice.pace.gatech.edu/ (VPN needed; use Chrome, Firefox or Edge)
VPN: GlobalProtect client from vpn.gatech.edu. Required off campus, on eduroam and in dorms. Not required on wired campus-building connections. Do not use the clientless browser VPN.
Docs index: https://gatech.service-now.com/home?id=kb_article_view&sysparm_article=KB0042503
Guest KB search: https://gatech.service-now.com/home?id=kb_search&query=PACE%20<topic>
Status: https://status.gatech.edu/
Status RSS: https://status.gatech.edu/pages/5be9af0e5638b904c2030699/rss
Status API (maintenance.active, maintenance.upcoming): https://api.status.io/1.0/status/5be9af0e5638b904c2030699
Announcements: https://pace.gatech.edu/pace-announcements-blog/
Paths [3P]: home /home/hice1/<gtuser>; ~/scratch -> /storage/ice1/<x>/<y>/<gtuser>; course shared /storage/ice-shared/<course>
Support: students go to their instructor or TA. Instructors and TAs use pace-support@oit.gatech.edu.

=== 2. LOGIN AND SSH (Andrew runs these and enters password and Duo personally)
# First login [PACE]
ssh <gtuser>@login-ice.pace.gatech.edu

# ~/.ssh/config on the Mac [3P]
Host ice
    HostName login-ice-1.pace.gatech.edu
    User <gtuser>
    ControlMaster auto
    ControlPath ~/.ssh/cm-%C
    ControlPersist 8h
    ServerAliveInterval 60
    ServerAliveCountMax 5

# Open and use the master connection
ssh ice true                 # Andrew completes password and any Duo prompt
ssh -O check ice             # confirms the master is alive
ssh ice 'hostname; pace-quota'   # should reuse the socket with no prompt
ssh -O exit ice              # closes it

# Optional key probe [3P]
ssh-keygen -t ed25519 -f ~/.ssh/ice_ed25519
ssh-copy-id -i ~/.ssh/ice_ed25519.pub <gtuser>@login-ice-1.pace.gatech.edu
ssh -o BatchMode=yes -o IdentitiesOnly=yes -o ControlPath=none -i ~/.ssh/ice_ed25519 <gtuser>@login-ice-1.pace.gatech.edu hostname
# A printed hostname means the key alone is enough. "Permission denied" means another factor is needed.
ssh -v -o ControlPath=none <gtuser>@login-ice-1.pace.gatech.edu true 2>&1 | grep -iE 'Authentications that can continue|Authenticated'

=== 3. FIRST CHECKS ON THE LOGIN NODE (light commands only)
pace-quota
ls -ld "$HOME" "$HOME/scratch" "$(readlink -f "$HOME/scratch")"
umask
module spider python
module spider anaconda3
module spider uv
module spider git
module spider ffmpeg
module spider google-cloud-sdk
module spider rclone
pace-check-queue ice-cpu
pace-check-queue ice-gpu

# Restrict the data directory before any download [ADAPTED]
mkdir -p ~/scratch/waymo && chmod 700 ~/scratch/waymo

=== 4. ~/.bashrc EXPORTS (set before any install)
export UV_CACHE_DIR=~/scratch/.uv_cache                # [PACE] ICE form of KB0045066
export UV_PYTHON_INSTALL_DIR=~/scratch/.uv_python      # [UPSTREAM] default ~/.local/share/uv/python
export PIP_CACHE_DIR=~/scratch/.cache/pip              # [UPSTREAM] default ~/.cache/pip
export APPTAINER_CACHEDIR=~/scratch/.apptainer/cache   # [UPSTREAM] default $HOME/.apptainer/cache
# Do not run conda init on PACE. To undo it: conda init --reverse

=== 5. ENVIRONMENT BUILD JOB [ADAPTED from KB0045066, Option B]
# Do not "module load python" alongside uv.
#!/bin/bash
#SBATCH -J create-env
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH -t 1:00:00
#SBATCH -o env-setup-%j.out

module load uv/0.9.17
cd ~/scratch/<project>
uv venv --python <X.Y> myenv
source myenv/bin/activate
uv pip install <packages>
uv pip freeze > requirements.txt

=== 6. INTERACTIVE JOBS
salloc -N1 --ntasks-per-node=4 -t1:00:00                      # [PACE] 4 cores, default 1 GB per core
salloc -N1 --ntasks=1 --cpus-per-task=8 --mem=32G -t2:00:00   # [ADAPTED]
salloc --gres=gpu:L40S:1 --ntasks-per-node=1 -t1:00:00        # [ADAPTED] from the PACE H100 example

=== 7. BATCH HEADERS (no -p and no -A on ICE)
# CPU job [ADAPTED from SlurmPythonExample in KB0042096]
#!/bin/bash
#SBATCH -Jagatha-stage1                    # Job name
#SBATCH -N1 --ntasks-per-node=1           # One node, one task
#SBATCH --cpus-per-task=8                 # Cores for the task
#SBATCH --mem-per-cpu=4G                  # Default is only 1 GB per core
#SBATCH --tmp=100G                        # Optional: reserve space in ${TMPDIR}
#SBATCH -t4:00:00                         # Limits: 18 h and 512 CPU-hours
#SBATCH -oReport-%j.out                   # Combined stdout and stderr
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=<gtuser>@gatech.edu
cd $SLURM_SUBMIT_DIR
module load uv/0.9.17
source ~/scratch/<project>/myenv/bin/activate
srun python <script>.py

# GPU job [ADAPTED from H100_example in KB0042096]
#!/bin/bash
#SBATCH -Jl40s-check                      # Job name
#SBATCH -N1 --ntasks-per-node=1           # One node, one task
#SBATCH --cpus-per-task=8                 # Cores for the task
#SBATCH --gres=gpu:L40S:1                 # GPU type and count per node (-G L40S:1 also works)
#SBATCH --mem-per-gpu=64G                 # About 95 GB per GPU is the node maximum (inference from 768 GB less 8 GB reserved)
#SBATCH -t1:00:00                         # Limits: 16 h and 16 GPU-hours
#SBATCH -oReport-%j.out                   # Combined stdout and stderr
cd $SLURM_SUBMIT_DIR
srun nvidia-smi

Submit and monitor [PACE]: sbatch <file>; squeue -u <gtuser>; scancel <jobid>; sacct -X -j <jobid>; pace-job-summary <jobid>

=== 8. OUTBOUND NETWORK TEST (inside a job, not on a login node)
salloc -N1 --ntasks=1 -t0:15:00
curl -sS -o /dev/null -m 20 -w '%{http_code} %{remote_ip}\n' https://storage.googleapis.com/
# Any HTTP status code means the node reached Google. A timeout or connection error means it did not.

=== 9. READ-ONLY LIMIT INSPECTION [UPSTREAM Slurm field names]
sacctmgr -P show qos format=Name,MaxWall,MaxTRESPerUser,MaxJobsPerUser,MaxSubmitJobsPerUser
scontrol show partition ice-cpu
scontrol show partition ice-gpu

=== 10. LIMITS AT A GLANCE
Login node, per user: 1 CPU-equivalent; 4 GB total memory (SIGKILL, exit 137, no message); 5000 open files.
Login node policy: no process at 90%+ of a CPU for over 30 minutes or at 20% of memory.
Compute node: 131072 open files.
Job limits: CPU 512 CPU-hours and 18 h; GPU 16 GPU-hours and 16 h.
Per user: 50 GPU jobs queued and running combined; 32 GPU-hours of running GPU work.
No-request default: 1 core, 1 GB per core, 1 hour.
Storage: home 30 GB, daily snapshot. Scratch 300 GB and 1 million files, no backup. Local ${TMPDIR} cleared at job end.
Retention: scratch files untouched for 120 days deleted at semester end. Home deleted after one year without access or login.
Partitions: ice-cpu, ice-gpu, coc-cpu, coc-gpu, pace-cpu, pace-gpu, coe-gpu. Assigned automatically.
QOS: pace-ice, coe-ice, coc-ice (optional). coc-grade, coe-grade, pace-grade (instructors and TAs only).
GPU requests: --gres=gpu:L40S:1 or -G L40S:1 or --gres=gpu:1 -C L40S. Also V100, RTX_6000, A40, A100, H100, H200, MI210, rtx_pro_6000_blackwell.
Constraints: -C intel, -C amd, -C graniterapids, -C localNVMe, -C localSAS, -C cuda12, -C HX00, -C 'L40S|HX00'.
Apptainer: compute nodes only. Spack: compute nodes only (module load spack).

=== 11. MAINTENANCE
Scheduled in November 2026: none found. Status API at 2026-11-01 20:35 UTC: active [] and upcoming [].
Next: Jan 11-13, 2027 and May 11-13, 2027, both tentative.
Recent: Jan 12-16, May 12-15, Aug 11-14 and Oct 5-8, 2026.