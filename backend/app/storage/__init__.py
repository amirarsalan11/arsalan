"""
Storage abstraction package.

Intentionally empty in Milestone 1.

This package will hold the single Storage Service that all upload/download
logic must go through, per the architecture rule:

    Application -> Storage Service -> S3 Compatible Provider

No boto3 dependency or S3/R2 client code is introduced yet — deferred until
the Storage implementation milestone.
"""
