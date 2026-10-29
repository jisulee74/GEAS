# Large/private data policy

Both repositories are public. Raw observations, facility memberships, externally supplied documents,
credentials and model binaries are kept in access-restricted workspace backups. Git LFS changes storage,
not repository confidentiality, and is not enabled for these private inputs.

GitHub blocks ordinary Git files larger than 100 MiB; LFS storage/bandwidth has account quotas and billing.
See [file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
and [LFS billing](https://docs.github.com/en/billing/concepts/product-billing/git-lfs).
No LFS migration/history rewrite, paid storage provisioning or public raw-data upload was performed.

The workspace is mounted from an external Lustre network filesystem, consistent with the administrator's
persistent `/NHNHOME/WORKSPACE/` instruction. This is outside the container root filesystem. A second copy
on the same Lustre filesystem does not protect against loss of that storage service.
An independent external account/bucket has not been connected or authorized; no offsite upload is claimed.

Historical Git commits remain unchanged. Removing a file from the current tree does not remove old copies
from history. Previously tracked credentials should be rotated. Private/external content previously present
in history needs separate repository-owner handling; force push is intentionally not used.
