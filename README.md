# dendrite_vldb

Artifacts for March 2022 Dendrite VLDB Submission

- Source code for Dendrite Log is in dendrite\_log\_src
- Source code for Dendrite-Pin is in dendrite\_pin\_src

Functionally, the code for these is the same except that Dendrite-Pin
has some customized thread-local data storage code. This is because Pin
does not support TLS coming from built-in language features.

Notebook with graph generation over high level experiment results is in
experiment\_results. Raw result files uploaded soon in same directory.
