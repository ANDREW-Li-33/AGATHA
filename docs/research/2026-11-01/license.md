# TOPIC license: What the Waymo Open Dataset license allows for this project, and what happens to data sent to Anthropic and Supabase (Waymo Dataset License Agreement for Non-Commercial Use, March 2025; pages read 2026-11-01). Not legal advice: this reports what the text says, where it is ambiguous, and who to ask.

## LIC-1 [high CRIT] Q1. What does Non-commercial Purposes cover?
CLAIM: Use is licensed only for Non-commercial Purposes: research, teaching, scientific publication and personal experimentation. Purposes mainly aimed at commercial advantage or pay are excluded, and so are litigation, licensing or enforcement purposes.
EVID: The definition lists the four purposes and adds benchmarking for academic or applied research publication. It excludes purposes "primarily intended for or directed towards commercial advantage or monetary compensation" (Waymo Dataset License, March 2025). Inference on grammar: the words about partial purpose attach to the litigation, licensing and enforcement limb, so the commercial limb is a primary-purpose test.
SRC: https://waymo.com/open/terms/
IMPACT: The classification turns on Andrew's real primary purpose, so they should be asked what it is.
VERDICT: confirmed

## LIC-2 [medium CRIT] Q1. Does preparing a portfolio project for job interviews fit?
CLAIM: The license and FAQ say nothing about portfolios, job seeking or employers. Inference: a genuine research prototype later shown to employers fits research or personal experimentation, but a project built mainly to win paid work could be read as directed at monetary compensation. Classify as ambiguous.
EVID: A term search of the license text alone found zero hits for portfolio, job, interview, recruit, hiring, employer, employee and student. The FAQ examples cover a university researcher and researchers at companies; none concerns job seeking.
SRC: https://waymo.com/open/terms/
IMPACT: Ask Andrew whether this is primarily research or primarily a job-search asset, and put the question to Waymo. Do not treat it as settled either way.
VERDICT: confirmed

## LIC-3 [medium CRIT] Q1. Does showing the project to a prospective employer, including Waymo itself, change anything?
CLAIM: There is no carve-out for showing work to Waymo or any employer. Inference: showing Waymo its own data carries little practical distribution risk, but the purpose test still applies and anything shared has no protection against Waymo.
EVID: Section 2 names no audience exception. In Section 5 the licensee agrees not to assert rights in Derivative IP against Waymo, its affiliates, successors or assignees. Section 6 brings in the Waymo Terms of Service, whose Section 6 treats communications about Waymo or its products as non-confidential and Waymo's property.
SRC: https://waymo.com/terms/
IMPACT: Ask Andrew which company the 2026-11-05 recruiter call is with. Treat a Waymo interviewer like any other recipient unless Waymo says otherwise in writing.
VERDICT: partially_correct
CORRECTED: There is no carve-out for showing work to Waymo or any employer, and the purpose test still applies. Inference: showing Waymo its own data carries little practical distribution risk. Protection against Waymo is limited, not absent: Section 5(a) stops Andrew asserting IP rights in Derivative IP against Waymo, and ToS Section 6 makes feedback about Waymo or its products non-confidential and Waymo's property, but whether an interview conversation counts as ToS feedback is unclear, and a separate NDA could change the position.
VERIF-EVID: Section 2 names no audience exception. Section 5(a) reaches only claims over rights protecting Derivative IP, and the 5(b) license activates only on breach of 5(a) or 4(b). ToS Section 6 covers information about Waymo or its products communicated to Waymo. The ToS governs websites and online services, and license Section 6 extends it to use of the Dataset. None of this supports the wording that anything shared has no protection.
VERIF-SRC: https://waymo.com/terms/

## LIC-4 [high] Q1. Are there limits on autonomous-vehicle work?
CLAIM: AV research is not barred. Section 4(a) forbids using the Dataset, or models and weights trained on it, in operating or assisting a vehicle, in Production Systems, or for other primarily commercial purposes. Inference: scripted replay of recorded trajectories in CARLA does not operate a vehicle, so the MVP scope fits.
EVID: The preamble says the release is to help research in machine perception and autonomous driving. The FAQ treats a company researcher experimenting on internal, non-customer systems as acceptable, and treats test-track use by an AV company, or deployment in a system for customers, as unacceptable.
SRC: https://waymo.com/open/faq/
IMPACT: Supports the design's scope boundary. Anything trained on the Dataset cannot move into an employer's vehicle or product work later.

## LIC-5 [medium CRIT] Q1 and Q4. Does a demo count as a Production System?
CLAIM: Inference: a publicly reachable Streamlit Scenario Explorer serving Dataset content would be a Production System, because the definition covers systems mainly meant to serve customers or other third parties, paid or not. Read literally, the definition could also reach Anthropic, Supabase or even PACE infrastructure, though the FAQ suggests the target is a licensee's own customer-facing deployment.
EVID: Section 4(a)(ii) bans use in Production Systems, defined to include infrastructure primarily meant to provide a product or service to customers or other third parties even without payment. FAQ Example 2 contrasts internal systems that do not serve customers. The FAQ also offers Google Cloud credits and treats university researchers as acceptable, which sits badly with the widest reading.
SRC: https://waymo.com/open/terms/
IMPACT: Keep Streamlit private (local or SSH tunnel) until cleared. Ask Andrew whether a public demo is planned, and include this in the Waymo questions.
VERDICT: partially_correct
CORRECTED: Section 4(a)(ii) bans use in Production Systems. Inference: a publicly reachable Streamlit Scenario Explorer serving Dataset content could be read as one, because the definition covers systems mainly meant to serve customers or other third parties, paid or not; would be is too strong, since a research demo may not be primarily intended to provide a service. The firmer bar is Section 2(b), because a public site distributes Dataset content to non-registrants whatever its Production System status. Read literally the definition could also reach Anthropic, Supabase or PACE, though the FAQ suggests the target is a licensee's own customer-facing deployment.
VERIF-EVID: Definition and Section 4(a)(ii) match the claim. FAQ Example 4, not cited, is the closest support for the narrow reading: deploying weights in a system intended for current or future customers. The Google Cloud point is overstated: the FAQ only says faculty and PhD students can apply to Google's research credit program, so it is not a general offer.
VERIF-SRC: https://waymo.com/open/terms/

## LIC-6 [high] Q1. Can results be used to judge vehicle performance?
CLAIM: Waymo's FAQ says the Dataset cannot be used to evaluate real-life vehicle performance.
EVID: The FAQ answers no, explaining that manually driven and autonomous data are mixed without labels, so no conclusion about a specific vehicle's behavior or performance is supported.
SRC: https://waymo.com/open/faq/
IMPACT: Incident outputs and portfolio text must describe scene events, not the Waymo Driver's performance. The design's scope wording already agrees; keep to it on the recruiter call.

## TP-1 [high CRIT] Q2. Is there language on service providers, cloud, AI services or authorized users?
CLAIM: No. The license never mentions service providers, vendors, processors, contractors, agents, cloud, hosting, AI services, sublicensing or transfer. It neither expressly permits nor expressly forbids sending Dataset content to the Claude API or Supabase.
EVID: The grant is a personal, non-exclusive license to use, reproduce and modify. Section 2 gives three sharing routes: small extracts in research publications, distribution to registrants, and publication of models. Section 8 reserves everything else. A term search of the license text returned zero hits for each of those words; the only third-party wording concerns Production Systems and recipients of Distributed WOD Models.
SRC: https://waymo.com/open/terms/
IMPACT: Nothing in the text clears Stage 2 or Stage 3 uploads. This is the core of the M0 rights gate.
VERDICT: confirmed

## TP-2 [medium CRIT] Q2. Are annotated frames and JSON covered, and is an upload distribution?
CLAIM: Inference: annotated frames, unannotated clips and SceneWindow JSON holding Waymo tracks are copies or modifications of the Dataset. If an upload is distribution, Section 2(b) blocks it because Anthropic and Supabase are not known registrants. Distribute is undefined, and that is the central ambiguity.
EVID: Dataset is defined to include any data in the Waymo Open Dataset. Section 2(b) allows distributing the Dataset, copies and modifications only to people who registered at waymo.com/open and accepted the Terms. The license does not define distribute and does not say whether a vendor acting for the licensee is a recipient.
SRC: https://waymo.com/open/terms/
IMPACT: The question covers JSON as well as images. Ask Waymo directly whether a processor acting for the licensee is a distribution recipient.
VERDICT: confirmed

## TP-3 [medium CRIT] Q2. Which way does the text lean on third-party services?
CLAIM: Inference: both readings have support. Treating a service as a tool Andrew uses is most defensible for transient, no-training processing, and weakest for stored copies or a service that may use the data itself.
EVID: For: the FAQ points users to Colab and Google Cloud credits, so third-party compute is contemplated (though Google shares a parent with Waymo, so the example may not generalize), and Anthropic's Commercial Terms treat inputs as confidential with no training. Against: the license is personal with no sublicense right, Section 8 reserves other rights, the FAQ stresses limits on distribution, and Supabase takes a license to reproduce, distribute and display data as needed to run its service.
SRC: https://waymo.com/open/faq/
IMPACT: Treat as neither cleared nor banned. Inference: if Waymo declines, an open-weight vision model on PACE GPUs would avoid the third-party question, at the cost of redesigning Stage 2.
VERDICT: partially_correct
CORRECTED: Inference: both readings have support. Treating a service as a tool Andrew uses is most defensible for transient, no-training processing, and weakest for stored copies or a service that may use the data itself. The FAQ support is thinner than stated: its Colab pointer is to tutorials it says use sample frames and do not access the actual dataset files, and only its Google Cloud credits answer speaks of working with the dataset, for faculty and PhD students. Supabase's rights are wider than stated: besides the service license it takes a perpetual, irrevocable license over Customer Data incorporated in Aggregated Data, while also promising not to train AI models on Customer Data.
VERIF-EVID: Read both FAQ answers directly. Anthropic Commercial Terms Sections B and E.1 verified: no training on Customer Content, and Customer Content is Customer's Confidential Information. Supabase Terms raw text shows both licenses and the no-training clause; Aggregated Data is defined as aggregate and anonymized use. The perpetual license is the strongest fact on the against side and was missing.
VERIF-SRC: https://waymo.com/open/faq/

## TP-4 [high CRIT] Q2. Who may access the data?
CLAIM: Copies may go only to people who have themselves registered and accepted the Terms. If Georgia Tech or another entity exercises the rights through Andrew's use, Andrew warrants they can bind it or that an authorized person has already registered. Inference: a student normally cannot give that warranty for Georgia Tech.
EVID: You means the individual or entity exercising the rights. The closing paragraph of the license sets the warranty and refers to the institution named at registration. GitHub issue 1014 shows the form collects an Institution field. Georgia Tech's Signature Authority page says only those with written delegation can bind it and that signing without authority may create personal liability.
SRC: https://generalcounsel.gatech.edu/legal-affairs/signature-authority
IMPACT: Settle this before registering. Ask Andrew whether this is personal, lab or course work and what they will enter as Institution. Collaborators must register themselves.
VERDICT: confirmed

## TP-5 [high CRIT] Q2. What do the Anthropic and Supabase contracts require of Andrew?
CLAIM: Both services make Andrew warrant holding the rights to submit the data and make Andrew answer for claims about it, so the Waymo ambiguity becomes Andrew's personal contractual exposure.
EVID: Anthropic Commercial Terms L.1 has the customer warrant all rights needed to submit Inputs, and K.2 has the customer defend and indemnify for claims related to Inputs. Consumer Terms Sections 4 and 11 are similar. Supabase Terms 9(a) has the customer warrant rights in Customer Data, and 10(b) adds an indemnity.
SRC: https://www.anthropic.com/legal/commercial-terms
IMPACT: Andrew should not make these warranties for Waymo content until they have a written answer from Waymo.
VERDICT: confirmed

## TP-6 [high] Q2. How does Supabase handle stored data?
CLAIM: Supabase takes a license to reproduce, distribute, use and display Customer Data as needed to provide the service, plus a perpetual license over data in anonymized Aggregated Data. It commits not to use Customer Data to train AI without consent. Its DPA deletion terms cover personal data only.
EVID: Terms 8(b) sets both licenses and the AI commitment. The DPA defines Covered Data around personal data, and clause 11.2 deletes it after a 30-day period following expiry. Storage docs say buckets are private by default, with access through RLS-checked JWTs or time-limited signed URLs, while a public bucket is open to anyone with the URL.
SRC: https://supabase.com/terms
IMPACT: A private bucket limits viewers, not the transfer itself. Metadata without imagery is the lowest-risk Supabase use. A signed URL handed to a recruiter would be distribution.

## STO-1 [high CRIT] Q3. Does the license impose storage or security obligations?
CLAIM: No. It sets no storage location, encryption, access-control or security requirement, so its text does not bar a shared university cluster.
EVID: A term search of the license text found zero hits for encrypt, secur, confidential, store, storage and retention. The only related duties are the Section 2(b) distribution limit and the Section 9 duty on termination. FAQ Example 1 accepts a university researcher's use but says nothing about infrastructure.
SRC: https://waymo.com/open/terms/
IMPACT: PACE is not blocked by the license and no license-driven encryption step is needed.
VERDICT: confirmed

## STO-2 [medium CRIT] Q3. What does a shared cluster mean for file permissions?
CLAIM: Inference: Dataset files readable by other cluster users could amount to making copies available to unregistered people, which Section 2(b) does not allow, so data and output directories should be private.
EVID: Section 2(b) limits recipients, and Section 9 treats access as something the licensee controls. Default permissions on ICE could not be confirmed from a live primary source.
SRC: https://waymo.com/open/terms/
IMPACT: Setup step: set private permissions before the first download, verify them, and avoid shared course directories.
VERDICT: confirmed

## STO-3 [medium CRIT] Q3. Is ICE itself suitable for this project?
CLAIM: PACE describes ICE as an educational resource GT faculty use to teach courses, so using it for a personal or portfolio project is a PACE question separate from the Waymo license. Current ICE quota, purge and permission rules could not be confirmed from a primary source.
EVID: The PACE ICE page and Get Started page describe it as an instructional cluster, with access requested by faculty for classes, and describe a separate research cluster. docs.pace.gatech.edu did not resolve when fetched.
SRC: https://pace.gatech.edu/ice-cluster/
IMPACT: Ask Andrew how they obtained ICE access and, if through a course, whether this project falls within it.
VERDICT: partially_correct
CORRECTED: PACE describes ICE as an instructional resource for classes and workshops, with instructors, not students, applying, so using it for a personal or portfolio project is a PACE question separate from the Waymo license. Current ICE storage rules are published in PACE's knowledge base: home 30 GB with daily snapshots, deleted one year after access or login stops; scratch 300 GB and 1M files, no backup, with files untouched for 120 days deleted at semester end; local job disk cleared at job end. Default file permissions remain unconfirmed.
VERIF-EVID: docs.pace.gatech.edu fails for me too, but pace.gatech.edu links its documentation to Georgia Tech's ServiceNow knowledge base, and those articles returned text without sign-in. KB0042094 (updated 2026-02-17) gives the storage rules. KB0042102 (2026-06-26) tells students not to file the ICE application. KB0042096 (2026-09-22) caps a job at 16 GPU hours and 16 hours GPU walltime, with 32 GPU-hours running at once. KB0042095 (2026-08-21) confirms 4 nodes with 8x L40S 48 GB. KB0042194 lists independent student research as a PACE use but labels ICE for academic classes and Phoenix as the research cluster. Scratch policy KB0042199 covers Phoenix and Hive, not ICE.
VERIF-SRC: https://gatech.service-now.com/api/sn_km_api/knowledge/articles/KB0042094

## PUB-1 [high CRIT] Q4. What attribution is required, and on what?
CLAIM: Every item of Derivative IP needs one of the two Section 1 attributions, and Derivative IP is broad enough to reach pipeline code, outputs and models. Inference: the sentence form states that use of the work is governed by the license, while the BibTeX form does not, so the choice matters if Andrew wants an open-source license on the repository.
EVID: Derivative IP covers derivative works, other works made or developed using the Dataset including software and models, and inventions. Section 1 offers a sentence-form notice or a BibTeX entry. The Section 1 BibTeX gives the year as 2019-2025, while the Citation block on the same page gives 2019.
SRC: https://waymo.com/open/terms/
IMPACT: Add the notice to the repository from the first commit and use the Section 1 form. The Section 5 non-assert also covers this work.
VERDICT: confirmed

## PUB-2 [high CRIT] Q4. What may be published from the data itself?
CLAIM: The only express right to publish Dataset content openly is for small extracts used as illustration in the licensee's research publications. Small and research publications are undefined, and the license never mentions images, videos, demos, blogs, theses or portfolios.
EVID: Section 2(a) gives the right. FAQ Example 1 is a paper with small extracts and attribution. A term search found zero hits for video, image, demo, blog, thesis and portfolio.
SRC: https://waymo.com/open/terms/
IMPACT: A paper or technical report is the safest vehicle. A portfolio page is not clearly a research publication, so ask Waymo.
VERDICT: confirmed

## PUB-3 [medium CRIT] Q4. Are demo videos on a portfolio or in an interview acceptable?
CLAIM: Inference: a public portfolio video of source footage or whole segments falls outside Sections 2(a) and 2(b). Sending a clip file to an unregistered recruiter falls outside 2(b). A live screen share with no file handed over is not clearly distribution, but it is not expressly permitted either.
EVID: The grant covers use, reproduction and modification, not display; Section 8 reserves other rights. Waymo ToS Section 3 bars public display of its Services unless expressly permitted, though whether the Dataset counts as Services is unclear.
SRC: https://waymo.com/open/terms/
IMPACT: For 2026-11-05, plan on code, architecture and aggregate results with no Dataset frames unless Waymo replies first. Ask Andrew what they meant to show.
VERDICT: confirmed

## PUB-4 [medium] Q4. What about trained models, meshes and CARLA renders?
CLAIM: Trained models may be published under conditions. Inference: a PINGS neural map or exported mesh may be a Distributed WOD Model or a Dataset modification, and CARLA renders are at least Derivative IP; the text does not settle which applies.
EVID: Section 2(c) allows publishing model architectures, weights and associated derivatives including new datasets, given a prominent notice, the license applying to recipients, a copy of the agreement, and attribution. Section 4(b) voids rights on non-compliant use.
SRC: https://waymo.com/open/terms/
IMPACT: A Stage 4 question with no effect on Stage 1. Include it in the Waymo questions.

## PUB-5 [high] Q4. May code and results be published?
CLAIM: Publishing results, and code containing no Dataset content, is supported with attribution.
EVID: FAQ Example 1 publishes results, Example 2 releases training code with attribution, and the MLPerf answers allow republishing benchmark scores. The FAQ's sample wording notes no endorsement is implied, and ToS Section 5 restricts use of Waymo marks.
SRC: https://waymo.com/open/faq/
IMPACT: A public repository is fine if it contains no Dataset content. Avoid wording that implies Waymo's endorsement.

## PUB-6 [high CRIT] Q4. What does this mean for repository fixtures?
CLAIM: Inference: real frames, Parquet rows or fixture files in a public repository would breach Section 2(b), so the fixture tests in design Section 10 must use synthetic data or stay out of version control.
EVID: Design Section 10 calls for fixture tests on a short segment. Waymo's repository holds only two v2 Parquet samples, camera_hkp and vehicle_pose, so the components Stage 1 needs are missing. Its LICENSE applies Apache 2.0 to the repository while the README says the Dataset is licensed separately, leaving those samples' status unclear.
SRC: https://github.com/waymo-research/waymo-open-dataset/tree/master/src/waymo_open_dataset/v2/testdata
IMPACT: Stage 1 design: generate synthetic Parquet fixtures from the public schema, and gitignore data and outputs before the first commit.
VERDICT: partially_correct
CORRECTED: Inference: real frames, Parquet rows or fixture files taken from downloaded Dataset content in a public repository would fall outside Section 2(b), so design Section 10 fixtures must be synthetic or stay out of version control. Waymo's repository holds only two v2 Parquet samples, camera_hkp and vehicle_pose. vehicle_pose is a Stage 1 input; camera_image, camera_calibration and lidar_box have no Parquet sample. The repository also holds v1 tfrecord samples. The root LICENSE applies Apache 2.0 except to src/waymo_open_dataset/wdl_limited, and the README says the Dataset is licensed separately, so the status of all these samples is unclear.
VERIF-EVID: GitHub tree API listing confirms the two Parquet files. Design Section 4 names vehicle_pose as an input, so the statement that the needed components are missing is too broad. The tree also lists two_frame.tfrecord and two other files under v2/perception/compat_v1/testdata, plus two tfrecords in tutorial/. Their sizes suggest real frames, but I did not open them. The testdata BUILD file tags the samples with the package's notice-type license metadata, which leans toward the repository license but is build metadata, not a grant. The researcher's Apache summary omitted the wdl_limited carve-out.
VERIF-SRC: https://github.com/waymo-research/waymo-open-dataset/tree/master/src/waymo_open_dataset/v2/testdata

## PRIV-1 [high] Q5. What are the restrictions on identifying people, blurred faces or plates?
CLAIM: The only privacy term in the license is Section 3, which bars use that violates applicable privacy laws. There is no express clause on re-identification or blurring. Waymo's blurring is a preparation step rather than a guarantee. Inference: frames should still be treated as possibly containing personal data.
EVID: The Privacy block describes a reasonable-care effort to remove or hide faces and plates and offers a form to request changes; the FAQ states more flatly that they are blurred. ToS Section 9 indemnity reaches violations of privacy rights. Anthropic's Usage Policy restricts obtaining biometric data, including through facial recognition, without permission.
SRC: https://waymo.com/open/terms/
IMPACT: No unblurring, plate reading or identity work. Stage 2 prompts should describe behavior only. Exclude any unblurred face or plate from demos and report it to Waymo.

## TERM-1 [high CRIT] Q6. What are the term, retention and deletion obligations?
CLAIM: There is no fixed term, expiry or routine deletion duty. Deletion applies on termination: rights end automatically on non-compliance, with no notice or cure step stated, and Andrew must remove access to and delete the Dataset.
EVID: Section 9 sets this out. A term search found no retention, expiry or survival language. Section 4(b) adds consequences for Distributed WOD Models, and Section 5(b) activates a license to Waymo in Derivative IP on breach of 5(a) or 4(b). Section 9 names only the Dataset, so its reach to derivatives is unclear.
SRC: https://waymo.com/open/terms/
IMPACT: Data may stay in place while compliant, subject to PACE rules. A breach could end the whole project, so keep an inventory of everywhere Dataset content sits.
VERDICT: confirmed

## TERM-2 [medium CRIT] Q6. Could external copies be deleted on demand?
CLAIM: Inference: if Section 9 applied, Andrew could not promptly delete data already sent externally, because deletion there is outside their control for a time.
EVID: API data is deleted within 30 days, or kept up to 2 years if flagged. Consumer training pipelines may hold data up to 5 years, and trained models keep what they absorbed. Supabase's DPA deletion covers personal data after a 30-day period.
SRC: https://privacy.claude.com/en/articles/7996866-how-long-do-you-store-my-organization-s-data
IMPACT: Supports sending nothing until Waymo answers, and sending the minimum afterwards.
VERDICT: partially_correct
CORRECTED: Inference: if Section 9 applied, Andrew could not delete data sent to Anthropic by his own action at once, but he is not limited to waiting. Commercial Terms E.4 requires Anthropic to destroy Confidential Information promptly on request, except legal retention and automated backups. On Supabase he controls deletion: deleting a project removes associated data including backups. Consumer training is the case that cannot be undone, because runs already started and trained models keep the data.
VERIF-EVID: API retention figures verified in the Privacy Center (2026-07-01). Commercial Terms E.1 makes Customer Content Confidential Information, and E.4 was overlooked; how it works in practice is untested. Supabase DPA clause 11.2 sets a 30-day period after the agreement expires and appears limited to personal data, so it does not limit the customer's own deletion. Supabase docs: Pro keeps 7 days of daily backups and project deletion removes backups. The consumer retention article confirms the training limit.
VERIF-SRC: https://www.anthropic.com/legal/commercial-terms

## ANT-1 [high CRIT] Q7(a). Is API data used for training?
CLAIM: Not by default. The exception is content the customer reports as feedback or a bug report, or otherwise chooses to share.
EVID: Commercial Terms Section B says Anthropic may not train models on Customer Content. The Privacy Center says the same by default and notes that reported feedback may be used for training. The Consumer Terms confirm API keys fall under the Commercial Terms.
SRC: https://www.anthropic.com/legal/commercial-terms
IMPACT: An API key is the least exposed Anthropic route for Stage 2 once cleared.
VERDICT: confirmed

## ANT-2 [high CRIT] Q7(a). How long is API data retained?
CLAIM: Standard API inputs and outputs are deleted within 30 days. Exceptions: content flagged under the Usage Policy (up to 2 years, classification scores up to 7), legal requirements, Files API (until deleted or expired) and Batch results (29 days). Zero retention needs an arrangement with Anthropic and is unavailable on consumer plans and, by default, on Covered Models (Fable 5, Fable 5.1, Mythos 5, Mythos 5.1).
EVID: Privacy Center and API retention docs. The docs page also says conversation content is not retained by default except on Covered Models, which appears to conflict with the Privacy Center's 30-day statement. The Covered Models article describes 30 days as applying to those models.
SRC: https://platform.claude.com/docs/en/manage-claude/api-and-data-retention
IMPACT: Plan on 30 days and state that figure to Waymo. Avoid the Files and Batch APIs for Stage 2 unless needed.
VERDICT: partially_correct
CORRECTED: Standard API inputs and outputs are deleted within 30 days. Exceptions: content flagged under the Usage Policy (up to 2 years, classification scores up to 7), legal requirements, Files API (until deleted or expired), Batch (29 days) and feedback reports (5 years). Zero retention needs an arrangement with Anthropic and is unavailable on consumer plans and, by default, on Covered Models (Fable 5, Fable 5.1, Mythos 5, Mythos 5.1). For Covered Models, state at least 30 days to Waymo, not 30 days.
VERIF-EVID: Figures verified in the Privacy Center and API docs, and the apparent docs conflict is real. Detail wrong in the claim: the Covered Models article (shown as updated September 1, 2026) gives a 30-day minimum, at least 30 days and then deletion, while a companion support article and the API docs say 30 days. Retained Covered Model data is assessed by automated safety systems, and human review happens only through a controlled path, such as when content is flagged. Stage 2's model is undecided, so the figure depends on that choice.
VERIF-SRC: https://support.claude.com/en/articles/15425695

## ANT-3 [high CRIT] Q7(b). What applies to Claude Code on a subscription plan?
CLAIM: On consumer plans (Free, Pro, Max, Ultra), Claude Code follows the account's model-improvement setting. On: data may be used for training and retained up to 5 years. Off: no future training and 30-day retention. Team and Enterprise plans are commercial: no training by default and 30-day standard retention.
EVID: The Claude Code data usage page sets out both groups. The consumer retention article says data may be held de-identified for up to 5 years in training pipelines.
SRC: https://code.claude.com/docs/en/data-usage
IMPACT: Ask Andrew which plan they use and how the setting is configured, before any real data reaches a machine running Claude Code.
VERDICT: partially_correct
CORRECTED: On consumer plans (Free, Pro, Max, Ultra), Claude Code follows the account's model-improvement setting. On: data may be used for training and retained up to 5 years. Off: no future training and 30-day retention. Team and Enterprise plans are commercial with no training by default, but their retention is not uniformly 30 days. The Claude Code page gives 30 days as standard, while the Privacy Center limits its 30-day statement to API users and says Claude for Work products keep chats and coding sessions in the product. Enterprise organizations can also hold local session transcripts for 6 years by default through the Compliance API.
VERIF-EVID: Consumer half verified on the data usage page and the consumer retention article. The Team and Enterprise half has a source conflict the researcher did not flag: the commercial retention article (2026-07-01) versus the data usage page. The API retention docs give the 6-year default for local session transcripts unless a finite organization period is set, and the compliance sessions page limits those endpoints to Claude Enterprise. Only API key use has a firm 30-day figure.
VERIF-SRC: https://code.claude.com/docs/en/data-usage

## ANT-4 [high CRIT] Q7(b). Does turning the consumer setting off fully stop training?
CLAIM: No. Flagged content and submitted feedback can still be used, and runs already started and models already trained keep earlier data.
EVID: Consumer Terms Section 4 and the Privacy Policy list the flagged-content and feedback exceptions. The model improvement settings article explains that the off switch does not reach training already under way or finished models.
SRC: https://www.anthropic.com/legal/consumer-terms
IMPACT: A consumer plan with the setting off is still weaker than an API key.
VERDICT: confirmed

## ANT-5 [high CRIT] Q7. Which setting governs which route?
CLAIM: The consumer setting does not apply to API keys. Commercial treatment applies when Claude Code runs on an API key or a Team or Enterprise plan. A Claude for Education account follows the university's agreement.
EVID: The Consumer Terms send API key use to the Commercial Terms. The authentication docs say Claude Code uses ANTHROPIC_API_KEY once approved. The Privacy Center covers Education accounts. Consumer Terms Section 3 bars automated access other than via an API key or with express permission.
SRC: https://code.claude.com/docs/en/authentication
IMPACT: Stage 2 must use an API key, not a subscription login. If Claude Code ever handles cleared Dataset content, run it on an API key.
VERDICT: partially_correct
CORRECTED: The consumer setting does not apply to API keys. Commercial treatment applies when Claude Code runs on an API key or a Team or Enterprise plan. A Claude for Education account follows the university's agreement. An API key only takes effect on surfaces that read it: the CLI, VS Code extension, Agent SDK and GitHub Actions. Claude Desktop and cloud sessions do not read ANTHROPIC_API_KEY and use OAuth, apart from Desktop sessions on a third-party inference configuration. So the instruction to run on an API key cannot be met by setting the key for Desktop.
VERIF-EVID: The Consumer Terms preamble, Section 3 automated-access exception and the Education article all verified. The authentication page confirms interactive sessions prompt once to approve the key and remember the answer, so declining leaves the subscription in use; /status shows the active credential. The same page contains the Desktop limit. This session's paths point to Claude Desktop, and the docs say a session has a scratchpad only on a claude.ai login, so this session is not on an API key.
VERIF-SRC: https://code.claude.com/docs/en/authentication

## ANT-6 [high CRIT] Q7. Does Claude Code send the contents of files and images it reads to Anthropic?
CLAIM: Yes. Claude Code runs locally, but whatever enters the conversation goes over the network to the model, including file contents, command output and images it reads. Running it on PACE does not keep data local.
EVID: The data usage page says it sends data over the network to reach the model. The how-it-works page says context holds file contents and command outputs. The tools reference says Read returns images as visual content, resized before sending.
SRC: https://code.claude.com/docs/en/tools-reference
IMPACT: Any real frame or row Claude Code reads or prints is a transfer to Anthropic.
VERDICT: confirmed

## ANT-7 [high CRIT] Q7. Are there other channels that carry content to Anthropic?
CLAIM: Yes. /feedback, /bug and /share send conversation history including code, retained 5 years, and may optionally create a GitHub issue in a public repository. Answering Yes to the survey follow-up uploads transcripts as-is, retained up to 6 months. Both can be disabled.
EVID: The data usage page describes each channel and its environment variables.
SRC: https://code.claude.com/docs/en/data-usage
IMPACT: Setup step: set DISABLE_FEEDBACK_COMMAND=1 and CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY=1 before real data is near a session.
VERDICT: confirmed

## ANT-8 [high CRIT] Q7. What does Claude Code keep locally?
CLAIM: Claude Code writes transcripts, including tool results, as plaintext under ~/.claude/projects/ for 30 days by default, protected only by OS file permissions.
EVID: The .claude directory docs and the data usage page describe the plaintext transcripts, cleanupPeriodDays and the lack of encryption at rest.
SRC: https://code.claude.com/docs/en/claude-directory
IMPACT: If run on PACE, transcripts sit in the PACE home directory. Keep ~/.claude private, lower cleanupPeriodDays, and count these files in the Section 9 inventory.
VERDICT: partially_correct
CORRECTED: Claude Code writes transcripts, including tool results, as plaintext under ~/.claude/projects/, protected only by OS file permissions. They are deleted after cleanupPeriodDays, default 30 and minimum 1, except that sessions started or last continued in Claude Desktop or Cowork are kept at any age unless desktopSessionCleanupPeriodDays is set (v2.1.248 or later). Dataset values can also persist outside transcripts: auto memory and history.jsonl are not removed by the age sweep, and file-history, paste-cache and the temp-directory scratchpad hold copies too.
VERIF-EVID: The .claude directory docs confirm plaintext storage and the 30-day default, but also carry the Desktop exemption, which the claim omits and which matters because this session appears to run in Claude Desktop. Lowering cleanupPeriodDays alone would not cover those sessions. Useful controls in the docs: CLAUDE_CODE_SKIP_PROMPT_HISTORY=1, claude purge and CLAUDE_CONFIG_DIR. On ICE, home has daily snapshots, so ~/.claude copies can outlast local deletion.
VERIF-SRC: https://code.claude.com/docs/en/claude-directory

## TOOL-1 [high CRIT] Q8. Is Claude Code inspecting annotated frames the same question as Stage 2?
CLAIM: Inference: yes. The license draws no line between a pipeline calling the API and a coding agent reading a file, so if Stage 2 needs clearance, so does this.
EVID: The license has no tool-based distinction (TP-1), and Claude Code sends what it reads to Anthropic (ANT-6).
SRC: https://waymo.com/open/terms/
IMPACT: Until Waymo answers, the Stage 1 acceptance check on box overlays should be done by Andrew's own eyes, not by Claude Code.
VERDICT: confirmed

## TOOL-2 [medium CRIT] Q8. Is the exposure limited to images?
CLAIM: Inference: no. Printed Parquet rows, box coordinates, tracks and error messages that echo values are Dataset data too.
EVID: Dataset includes any data in the Waymo Open Dataset, and Claude Code's context includes command outputs. Column names come from Waymo's public repository, not from licensed values.
SRC: https://waymo.com/open/terms/
IMPACT: Debugging with df.head() on real data inside Claude Code is the same transfer. Scripts should report only shapes, counts, schema and pass or fail.
VERDICT: confirmed

## TOOL-3 [medium CRIT] Q8. Which configuration is worst?
CLAIM: Inference: a consumer account with model improvement on is the one setup clearly outside the license, because it lets a non-registrant use the content for commercial model training, and Andrew would warrant rights they do not hold.
EVID: Consumer Terms Section 4 allows training unless opted out and requires a rights warranty. License Sections 2(b), 4(a)(iii) and 8 give no basis for that use. The license does not name this scenario, so this is a reading, not express text.
SRC: https://www.anthropic.com/legal/consumer-terms
IMPACT: Andrew should confirm this setting before any real data is on a machine where Claude Code runs.
VERDICT: partially_correct
CORRECTED: Inference: among Claude routes, a consumer account with model improvement on is the worst, because it lets a non-registrant use the content for commercial model training. This is a reading, not express text, so clearly outside overstates it. The rights warranty does not distinguish this setup, because every route requires one: Consumer Terms Section 4, Commercial Terms L.1 and Supabase 9(a). The off setting is not a clean contrast either, since flagged and feedback material stays trainable.
VERIF-EVID: Consumer Terms Section 4 allows training unless the user opts out and contains the warranty. License Sections 2(b), 4(a)(iii) and 8 give no basis for vendor training, but none names this scenario. The claim's own evidence concedes it is a reading, which sits badly with the words clearly outside.
VERIF-SRC: https://www.anthropic.com/legal/consumer-terms

## TOOL-4 [medium CRIT] Q8. Can Stage 1 proceed without any transfer?
CLAIM: Inference: yes. Claude Code can write code against public schemas and synthetic fixtures, Andrew runs it on real data and inspects overlays, and only aggregate metrics come back. No Waymo data has been downloaded yet, so this can be set up before any exposure.
EVID: Design Section 4 accepts a manually inspected frame sample. The Parquet schema is in Waymo's public repository and tutorial.
SRC: https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_v2.ipynb
IMPACT: The interim rule to put to Andrew: no real Waymo values or imagery in any Claude session until Waymo answers. Stage 1 need not wait for that reply.
VERDICT: partially_correct
CORRECTED: Inference: yes for third-party exposure. Claude Code can write code against public schemas and synthetic fixtures, Andrew runs it on real data and inspects overlays, and only aggregate metrics come back, so Stage 1 need not wait for Waymo's answer on vendors. It cannot skip the other gates. Downloading or using the Dataset is itself acceptance and is licensed only for Non-commercial Purposes, so Andrew's primary purpose (LIC-2), the institution question (TP-4) and ICE eligibility (STO-3) must be settled first. If the purpose question needs Waymo's answer, Stage 1 does wait.
VERIF-EVID: The license preamble makes downloading or using the Dataset an agreement to the Terms. The repository has CameraImageComponent, CameraCalibrationComponent, LiDARBoxComponent and VehiclePoseComponent, and tutorial_v2 shows their use. The claim's impact conflicts with LIC-2, which says not to treat purpose as settled. On no data yet: I found no Parquet, tfrecord or JPEG in the scratchpad, but I could not check PACE or Andrew's machine.
VERIF-SRC: https://waymo.com/open/terms/

## TOOL-5 [high CRIT] Q8. Can tooling enforce that rule?
CLAIM: Only partly. Read deny rules miss arbitrary subprocesses such as a Python script, and sandbox denyRead covers Bash commands but not the built-in Read tool. On Linux the sandbox needs bubblewrap and socat, and without failIfUnavailable it silently runs unsandboxed.
EVID: The permissions docs limit Read and Edit rules to built-in tools and recognized file commands. The sandboxing and settings docs describe denyRead, the Linux dependencies and sandbox.failIfUnavailable. Whether PACE provides those tools is unknown.
SRC: https://code.claude.com/docs/en/permissions
IMPACT: Use both controls plus workflow discipline, and keep real data outside the project working directory.
VERDICT: confirmed

## CON-1 [high CRIT] Q9. How does Andrew ask Waymo for clarification?
CLAIM: Waymo's named contact for license questions and commercial licensing is open-dataset@waymo.com. Inference: a recruiter's remark is not a license interpretation, so a written reply from that address is what counts. Reply time is unknown.
EVID: License Section 8 and the FAQ both name that address. The ToS lists legal@waymo.com for questions about the ToS. My GitHub issue search found no license-interpretation thread, though Waymo collaborators do answer issues there.
SRC: https://waymo.com/open/faq/
IMPACT: Send the questions this week, and plan 2026-11-05 as uncleared.
VERDICT: partially_correct
CORRECTED: Waymo's named contact for license questions and commercial licensing is open-dataset@waymo.com. Inference: a recruiter's remark is not a license interpretation, so a written reply from that address is what counts. Reply time is unknown. GitHub does hold license-related threads: in issue 66 a Waymo engineer answered a licensing question on the camera-model code and invited use-case requests. None I found interprets the Dataset license's distribution terms.
VERIF-EVID: Section 8, the FAQ and the ToS legal@waymo.com line all verified. The GitHub search statement is too broad: searching license returned 40 items. Issue 66 is a licensing exchange: on 2019-12-30 Pei Sun said the camera license and patent grant were the only ones available, were consistent with processing the Dataset in compliance with its license, and that other licenses were under discussion. His GitHub profile now lists Google DeepMind. In issue 911 a collaborator sends access trouble to the same address. Issue 389 is covered under PUB-3.
VERIF-SRC: https://github.com/waymo-research/waymo-open-dataset/issues/66

## CON-2 [medium] Q9. What care is needed in that email?
CLAIM: Inference: anything Andrew writes to Waymo about Waymo or its products may fall under ToS Section 6, which makes it non-confidential and Waymo's property, so the email should describe the use case without unpublished ideas.
EVID: ToS Section 6 covers questions, ideas and other information about Waymo or its products communicated to Waymo. Whether a license question qualifies is a reading of that clause.
SRC: https://waymo.com/terms/
IMPACT: Keep the email to data flows and purposes.

## CON-3 [high] Q9. Who can help at Georgia Tech?
CLAIM: Georgia Tech's legal office advises the Institute and does not advise individuals on personal matters. Research data use agreements go through the Office of Exchange Agreements in the Office of Sponsored Programs. Legal Affairs is reachable at asklegal@gatech.edu.
EVID: The Office of General Counsel FAQ says this. The page does not address licenses accepted by individual students.
SRC: https://generalcounsel.gatech.edu/all-faq
IMPACT: These routes are useful if the work is lab or course work. If it is personal, Waymo is the main route.

## WDL-1 [high] Related to Q2 and Stage 1. Does Waymo's camera-model code carry its own terms?
CLAIM: Code under wdl_limited/camera, which includes world_to_image, has a BSD license plus a patent grant that applies only when processing Waymo Open Dataset data in compliance with the Dataset license. The rest of the repository is Apache 2.0.
EVID: The README and the PATENTS file in that folder state this.
SRC: https://raw.githubusercontent.com/waymo-research/waymo-open-dataset/master/src/waymo_open_dataset/wdl_limited/camera/PATENTS
IMPACT: Fine for Stage 1 on Waymo data. Review before reusing that code on other data.

# VERIFIER MISSED
- Premise check: impl.md and the relayed request never mention a portfolio, job interview, recruiter or a 2026-11-05 call, and the design calls this a research prototype. LIC-2, LIC-3, PUB-3 and CON-1 rest on those premises, so confirm them with Andrew rather than carrying them forward.
- Georgia Tech policy layer: the Acceptable Use Policy requires IT resources to be used in line with applicable licenses and contracts and their intended use, and bars employees, including students acting as employees, from unauthorized use for commercial purposes or personal gain. A Waymo license breach on ICE is therefore also a GT policy issue, and a job-search purpose could be a problem on ICE even if Waymo accepts it. Source: https://policylibrary.gatech.edu/information-technology/acceptable-use-policy
- ICE data placement: keep Dataset files on scratch, not home, because home gets daily snapshots, which undercuts prompt deletion under Section 9. Scratch has no backup and loses files untouched for 120 days at semester end, so plan for re-download. Home files linger up to a year after ICE access ends, so delete before a course ends. GPU jobs stop at 16 hours, which matters for PINGS. Source: KB0042094 and KB0042096.
- Stage 1 code licensing: Waymo's camera ops in src/waymo_open_dataset/wdl_limited/camera are BSD-3 plus a limited patent grant, not Apache 2.0. The grant applies only to Waymo's own code used to process the Dataset in compliance with its license, and excludes claims infringed only through further modification. A NumPy restatement of camera_model.cc, like /private/tmp/claude-501/-Users-andrewli-Library-Application-Support-Claude-scratch-workspaces-2a1158d9-8d2c-4106-8450-10ec010ec513-2890a3af-9790-4d79-9c25-15cc7f06f616-scratch-2026-10-23-4d493d/58822fd1-3d5b-4902-bdc7-bc55c3a3b06c/scratchpad/wod_cam.py, is not that code, so the grant would not cover it. The cited patent, US 10,939,057, is titled Synchronized spinning LIDAR and rolling shutter camera system. I did not read its claims, so whether software projection is affected is unknown. Decide the projection approach before writing it, and note that a breach of the Dataset license also removes the grant's basis.
- The FAQ answers no to using the Dataset to evaluate real-life vehicle performance, because it mixes manual and autonomous driving without labels. The design's scope boundary fits, but incident reports, demo narration and interview statements must not imply conclusions about any real vehicle or the Waymo Driver.
- ToS Section 6 likely reaches the questions Andrew sends to open-dataset@waymo.com: questions, ideas and works about Waymo products become non-confidential and Waymo's property. Keep the email to a description of use, without unpublished design detail. ToS Section 5 also bars using Waymo marks without written permission, so avoid logos on a portfolio; the FAQ's MLPerf example adds a no-endorsement line.
- The FAQ's MLPerf answer is the only Waymo text accepting publication of results outside papers, including blogs and marketing materials. It covers MLPerf only, but it supports the aggregate-results plan for an interview and is a useful precedent to cite when asking Waymo about aggregate pipeline metrics.
- Possible publication route for reconstruction outputs: Section 2(c) allows publishing trained models and associated derivatives, including generated datasets, with a prominent notice, a copy of the Agreement and attribution. PINGS optimizes a neural map on Dataset input, so it may count. This is an inference; add it to the Waymo questions, since CARLA renders may be a cleaner demo than source frames.
- Vendor alternative: Anthropic's retention docs say that on Amazon Bedrock and Google-managed Claude on Google Cloud the cloud provider is the data processor. Given the FAQ's Google Cloud credits answer, ask Waymo about Google Cloud separately. This does not remove the third-party question.
- Sandbox enforcement gap for TOOL-5: Claude can retry a blocked command outside the sandbox through dangerouslyDisableSandbox unless sandbox.allowUnsandboxedCommands is false, and failIfUnavailable only matters with sandbox.enabled true. The docs imply Read deny paths merge into the sandbox configuration but do not say so outright. Settings bwrapPath and socatPath exist for custom binaries. A saved PACE software list shows neither tool, which says nothing about OS packages or user namespace support.
- Another channel missing from ANT-7: Remote Control stores the session transcript on Anthropic servers while connected, and this session is set up so it can be followed from another device. The feedback environment variables do not cover that. Check it, along with the account plan, before real data is near a session.
- Registration status: a Waymo collaborator said on 2026-10-20 that the form failure in issue 1014 was fixed, but issue 1015, reporting the same error, remains open with no replies. If registration fails, use open-dataset@waymo.com.

# UNRESOLVED
- Whether sending Dataset content to a service acting for the licensee (Claude API, Claude Code, Supabase) is distribution or permitted use. -> Written reply from open-dataset@waymo.com, stating the data flow, the 30-day retention and the no-training terms.
- Whether a research project also used in job applications stays within Non-commercial Purposes, and whether a demo to Waymo itself is treated differently. -> Andrew states the real primary purpose, then Waymo answers in writing.
- Whether a portfolio page, blog or thesis is a research publication, how small a small extract is, and whether a live interview screen share is acceptable. -> Written reply from open-dataset@waymo.com. Until then, keep Dataset frames out of portfolio pages and interviews.
- How far the Production Systems definition reaches: third-party infrastructure, shared university clusters, or a demo restricted to registered users. -> Written reply from open-dataset@waymo.com. Until then, keep Streamlit private.
- Pages not read: waymo.com/open/licensing/, /open/download/ and /open/data/perception/ redirect to Google sign-in, and I did not sign in. They may carry further terms. -> Andrew reads the license text shown at registration and checks it is the March 2025 version. A registration bug was reported fixed on 2026-10-20 in GitHub issue 1014.
- Andrew's Claude account type and model-improvement setting. -> Ask Andrew. They can check claude.ai/settings/data-privacy-controls.
- Andrew's capacity (personal, lab or course), the Institution value they will enter, and how they obtained ICE access. -> Ask Andrew before registration. If the work is lab or course work, ask the advisor or instructor, and the Office of Exchange Agreements or Legal Affairs.
- Which company the 2026-11-05 recruiter call is with and what Andrew planned to show. -> Ask Andrew.
- Current ICE quota, purge, backup and default-permission rules. docs.pace.gatech.edu did not resolve and the ServiceNow knowledge base was not read. A 2023 ICE paper surfaced by search suggested 15 GB home, 100 GB scratch, no scratch backup and semester-end deletion of untouched scratch, but I did not verify this. -> Andrew checks from a PACE shell and asks PACE support. Whether ICE provides bubblewrap and socat can be checked the same way.
- Apparent conflict in Anthropic's sources: the API retention docs say conversation content is not retained by default except on Covered Models, while the Privacy Center gives deletion within 30 days. -> Plan on 30 days. Ask Anthropic support only if the difference becomes decisive.
- Whether Waymo's public repository samples (v2/testdata Parquet, tutorial/frames) fall under Apache 2.0 or the Dataset license, and whether Section 9 deletion reaches derivative outputs. -> Include both in the Waymo email. Until then treat the samples as Dataset data and derivatives as covered.

# EXACT REFERENCE
All clause text below is paraphrased; read the originals at the cited URLs. Not legal advice.

WAYMO LICENSE, March 2025 (https://waymo.com/open/terms/)
Numbering: Sections 1 to 9 are the top-level list items, confirmed by the license's own cross-references to Sections 1, 4, 4(a), 4(b), 5(a) and 5(b). Sub-labels 2(a) to (c) and 2(c)(i) to (iv) are inferred from list nesting.
- Grant: personal, non-exclusive, royalty-free right to use, reproduce and modify for Non-commercial Purposes only.
- Definitions: Dataset includes any data in it. Derivative IP covers derivative works, other works made using the Dataset, and inventions. Production Systems covers infrastructure serving customers or other third parties, paid or not.
- 1 Attribution: one of two forms in any Derivative IP.
- 2(a) De Minimis: small extracts for illustration in Your research publications.
- 2(b) Community Distribution: Dataset, copies and modifications only to registrants who accepted the Terms, plus a promise not to assert claims over the distributed work.
- 2(c) Models: may publish models, weights and derived datasets with (i) prominent notice, (ii) the license binding recipients, (iii) a copy of the agreement, (iv) attribution.
- 3 Privacy: no use that violates applicable privacy laws.
- 4(a) Bars use in or assisting vehicle operation, in Production Systems, or for other primarily commercial purposes.
- 4(b) Non-compliant use of Distributed WOD Models voids rights, ends recipient rights, requires recipients to delete, and triggers the 5(b) license.
- 5 Non-assert: no claims over your Derivative IP against Waymo and affiliates; a license to Waymo takes effect on breach of 5(a) or 4(b).
- 6 Website Terms: Waymo ToS applies, including Sections 9, 10 and 11.
- 7 Disclaimer and liability limit.
- 8 All other rights reserved; contact open-dataset@waymo.com for commercial licensing or other questions.
- 9 Non-compliance ends rights automatically; remove access to and delete the Dataset.
- Closing paragraph (unnumbered): if the named institution or another entity exercises the rights, you warrant authority or that an authorized person already registered.

WAYMO ToS, effective Feb 20, 2018 (https://waymo.com/terms/)
- 3: no public display of the Services unless expressly permitted.
- 5: Waymo marks need written permission.
- 6: Feedback is non-confidential and Waymo's property.
- 9: indemnity, including for violation of others' IP or privacy rights.
- 14: California law, Santa Clara County venue.

WAYMO FAQ (https://waymo.com/open/faq/)
- Not to be used for evaluating real-life vehicle performance.
- Not an open source license.
- Examples 1 and 2 acceptable; Examples 3 and 4 unacceptable.
- Contact: open-dataset@waymo.com.

ANTHROPIC
- Commercial Terms (shown effective Jun 17, 2025): B no training on Customer Content; C DPA applies; D.2 Usage Policy; E Customer Content is confidential; K.2 customer indemnity for Inputs; L.1 customer warrants rights to submit Inputs.
- DPA H: deletion within 30 days after termination, with exceptions.
- Consumer Terms (shown effective Oct 8, 2025): 3 automated access only via API key or express permission; 4 training unless opted out, with flagged-content and feedback exceptions, and a rights warranty; 11 indemnity.

SUPABASE (https://supabase.com/terms)
- 1(a) and 2(f) Aggregated Data; 2(a) internal business purposes; 7(b) customer responsible for its data and security; 8(b) licenses and no AI training without consent; 9(a) rights warranty; 10(b) indemnity.
- DPA (Version 1, Aug 1, 2026): Covered Data is personal data; 11.2 deletion after a 30-day period following expiry.

ACTIVITY CLASSIFICATION (my reading of the text)
CLEARLY ALLOWED, assuming registration and a non-commercial purpose:
- Download to Andrew's own private storage and run local Stage 1 processing: joins, projection, wireframes, manifests, JSON.
- Andrew personally viewing frames.
- Sharing copies with a collaborator who has registered and accepted the Terms.
- Publishing code with no Dataset content, with attribution.
- Publishing aggregate metrics, with attribution.
- Small illustrative extracts in a paper or research report, with attribution.
- Describing the project and results verbally in an interview.
AMBIGUOUS, ask Waymo:
- The portfolio purpose itself.
- Stage 2 frames and JSON through the Claude API.
- Claude Code reading real frames or rows on an API key, on Team or Enterprise, or on a consumer plan with training off (worst of these).
- Clips in private Supabase storage.
- IncidentReport metadata in Supabase (lower risk).
- Live screen share of frames in an interview, including with Waymo staff.
- Small annotated extracts on a portfolio page or blog.
- CARLA replay video on a portfolio.
- Publishing a PINGS mesh.
- Streamlit gated to registered users.
- Keeping the data on ICE (license fine, ICE eligibility is PACE's question).
CLEARLY PROHIBITED on the plain text, or by the FAQ where marked:
- Use in or assisting vehicle operation, in Production Systems, or in an employer's product.
- Public Streamlit exposing Dataset content (2(b), 4(a)(ii)).
- Real frames, Parquet rows or tracks in a public repository or public bucket (2(b)).
- Public web video of source footage or whole segments (2(b); not a 2(a) research publication).
- Emailing clips or frames, or sending signed URLs, to an unregistered recruiter (2(b)).
- Litigation, licensing or enforcement purposes, even in part.
- Claiming results measure a real vehicle's performance (FAQ).
- Consumer plan with model improvement on and real Dataset content (inference: the license does not name this case; see TOOL-3).

SETUP BEFORE FIRST DOWNLOAD
PACE shell (paths are placeholders):
umask 077
mkdir -p "$WOD_DATA" "$WOD_OUTPUTS" && chmod 700 "$WOD_DATA" "$WOD_OUTPUTS"
ls -ld "$WOD_DATA" "$WOD_OUTPUTS" ~/.claude

.gitignore:
data/
outputs/
*.parquet
*.tfrecord

Claude Code settings.json (validate with claude doctor; absolute paths use the // prefix in Read rules):
{
  "cleanupPeriodDays": 7,
  "env": {
    "DISABLE_FEEDBACK_COMMAND": "1",
    "CLAUDE_CODE_DISABLE_FEEDBACK_SURVEY": "1"
  },
  "permissions": {
    "deny": ["Read(//abs/path/to/waymo_data/**)", "Read(//abs/path/to/waymo_outputs/**)"]
  },
  "sandbox": {
    "enabled": true,
    "failIfUnavailable": true,
    "filesystem": { "denyRead": ["/abs/path/to/waymo_data", "/abs/path/to/waymo_outputs"] }
  }
}
Limits: Read deny rules do not stop subprocesses, denyRead does not stop the Read tool, and failIfUnavailable stops Claude Code from starting on PACE if bubblewrap or socat is missing. Alternative broad switch: CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1.

Attribution, BibTeX form from Section 1 (citation record; copy the sentence form from Section 1 if preferred):
@misc{waymo_open_dataset, title = {Waymo Open Dataset: An autonomous driving dataset}, website = {\url{https://www.waymo.com/open}}, year = {2019-2025}}

QUESTIONS FOR open-dataset@waymo.com (keep to data flows and purposes)
1. May a licensee send annotated frames and label-derived JSON to a commercial LLM API acting only for the licensee, with no training on the data and deletion within 30 days? Would flagged content kept up to 2 years change the answer?
2. Is the same true of an AI coding assistant reading those files?
3. May the licensee keep short clips in private cloud storage (Supabase) accessible only to themself?
4. Does a university research project also shown in job applications remain Non-commercial?
5. Does a portfolio page, blog or thesis count as a research publication under 2(a), and what size counts as small? Are live demos in an interview acceptable?
6. May CARLA renders or a reconstructed mesh be shown publicly?
7. Does Production Systems reach third-party infrastructure, or a login-gated demo?
8. Are the repository samples under Apache 2.0 or the Dataset license?

PROCESS NOTE
I briefly saved three Anthropic pages as text under the scratchpad to search them, against the print-to-stdout rule. Those files are deleted. The harness also auto-saved some long outputs under ~/.claude/projects. No datasets were downloaded, nothing was signed in to, and PACE was not contacted.