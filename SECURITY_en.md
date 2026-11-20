# Security Statement

## Recommended Running User

For security purposes, do not use root or other administrator accounts to execute any commands. Follow the principle of least privilege.

## File Permission Control

- Set the system umask value on the host (including the host machine) and in containers to 0027 or higher. This ensures newly created folders have a maximum permission of 750 and newly created files have a maximum permission of 640.
- Apply access control and other security measures to sensitive content such as personal privacy data, commercial assets, source files, and various files saved during operator development. For example, for installation directory permission control and input public data file permission control in this project, refer to [Appendix A - Recommended Maximum Permissions for File/Folder Scenarios](#appendix-a---recommended-maximum-permissions-for-filefolder-scenarios).
- Apply access control during installation and use. Refer to [Appendix A - Recommended Maximum Permissions for File/Folder Scenarios](#appendix-a---recommended-maximum-permissions-for-filefolder-scenarios) for permission settings.

## Build Security Statement

When compiling and installing this project from source, you must compile it yourself. The compilation process generates intermediate files. Apply access control to these intermediate files after compilation to ensure file security.

## Runtime Security Statement

When a runtime error occurs, the process exits and prints error information. Use the error message to identify the specific cause.

## Public Network Address Statement

The public network addresses in this project code are as follows:

| Type | Open Source Code Address | Filename | Public IP Address/URL/Domain/Email/Archive Address | Description |
| :---: | :---: | :---: | :---: | :---: |
| Dependency | N/A | cmake/nlohmann_json.cmake | https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip | Download JSON source code from GitCode as a compilation dependency |
| Dependency | N/A | cmake/tests/gtest.cmake | https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz | Download GoogleTest source code from GitCode as a test dependency |
| Dependency | N/A | cmake/tensorflow.cmake | https://github.com/tensorflow/tensorflow/archive/v1.15.0.zip | Download TensorFlow 1.15.0 source code as a compilation dependency |
| Dependency | N/A | cmake/secure_c.cmake | https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.16.tar.gz | Download SecureC source code as a compilation dependency |

## Vulnerability Management

[Vulnerability Management](https://gitcode.com/cann/community/blob/master/security/security.md)

## Appendix

### Appendix A - Recommended Maximum Permissions for File/Folder Scenarios

| Type | Recommended Maximum Linux Permission |
| --- | --- |
| User home directory | 750 (rwxr-x---) |
| Program files (including scripts, libraries, etc.) | 550 (r-xr-x---) |
| Program file directory | 550 (r-xr-x---) |
| Configuration files | 640 (rw-r-----) |
| Configuration file directory | 750 (rwxr-x---) |
| Log files (completed or archived) | 440 (r--r-----) |
| Log files (active logging) | 640 (rw-r-----) |
| Log file directory | 750 (rwxr-x---) |
| Debug files | 640 (rw-r-----) |
| Debug file directory | 750 (rwxr-x---) |
| Temporary file directory | 750 (rwxr-x---) |
| Maintenance and upgrade file directory | 770 (rwxrwx---) |
| Business data files | 640 (rw-r-----) |
| Business data file directory | 750 (rwxr-x---) |
| Key components, private keys, certificates, encrypted file directory | 700 (rwx------) |
| Key components, private keys, certificates, encrypted files | 600 (rw-------) |
| Encryption/decryption interfaces and scripts | 500 (r-x------) |
