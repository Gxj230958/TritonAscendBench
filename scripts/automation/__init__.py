"""INTERNAL bench-team automation -- NOT part of bench release.

This directory is export-ignored from release tarballs and Docker images.
bench core code MUST NOT import from this module.

Layout:
    scripts/automation/
        lib/        -- shared library modules (atomic, queue, lock, ...)
        prompts/    -- LLM prompt templates
        spike/      -- one-off spike scripts (M0 GPFS / LLM format A/B)

Release boundary is enforced by:
    .gitattributes               -- ``export-ignore`` rules
    .dockerignore                -- Docker build context exclusion
    .github/workflows/ci.yml     -- release tarball smoke + bench core grep
"""
