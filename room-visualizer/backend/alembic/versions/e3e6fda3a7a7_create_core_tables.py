"""create core tables

Revision ID: e3e6fda3a7a7
Revises:
Create Date: 2026-09-07

Creates the six core Milestone 2 tables: tenants, api_keys,
allowed_domains, materials, render_jobs, usage_records.

This migration was hand-written rather than produced by
`alembic revision --autogenerate` because no live PostgreSQL instance
was reachable in the authoring environment. It was written to mirror
the ORM models in app/db/models/ exactly (column types, nullability,
defaults, FKs, indexes, unique constraints). Before relying on this in
a real environment, run it once against a real Postgres instance and
confirm `alembic check` / a diff against autogenerate reports no drift.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "e3e6fda3a7a7"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None


def upgrade() -> None:
    # gen_random_uuid() is provided by pgcrypto on Postgres < 14, and is
    # built in natively on Postgres 14+. Enabling the extension
    # defensively guarantees availability regardless of the exact
    # Postgres version, without any harm if it's already built in.
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto"')

    # --- tenants -----------------------------------------------------
    op.create_table(
        "tenants",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("slug", sa.String(length=100), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.UniqueConstraint("slug", name="uq_tenants_slug"),
    )
    op.create_index("ix_tenants_slug", "tenants", ["slug"], unique=True)

    # --- api_keys ------------------------------------------------------
    op.create_table(
        "api_keys",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("key_hash", sa.String(length=255), nullable=False),
        sa.Column("label", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_api_keys_tenant_id_tenants", ondelete="RESTRICT"
        ),
        sa.UniqueConstraint("key_hash", name="uq_api_keys_key_hash"),
    )
    op.create_index("ix_api_keys_tenant_id", "api_keys", ["tenant_id"])
    op.create_index("ix_api_keys_key_hash", "api_keys", ["key_hash"], unique=True)

    # --- allowed_domains ------------------------------------------------
    op.create_table(
        "allowed_domains",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("domain", sa.String(length=255), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_allowed_domains_tenant_id_tenants",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "tenant_id", "domain", name="uq_allowed_domains_tenant_domain"
        ),
    )
    op.create_index("ix_allowed_domains_tenant_id", "allowed_domains", ["tenant_id"])

    # --- materials -------------------------------------------------------
    op.create_table(
        "materials",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("texture_url", sa.String(length=1024), nullable=False),
        sa.Column("category", sa.String(length=100), nullable=False),
        sa.Column("width_mm", sa.Integer(), nullable=True),
        sa.Column("height_mm", sa.Integer(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_materials_tenant_id_tenants", ondelete="RESTRICT"
        ),
    )
    op.create_index("ix_materials_tenant_id", "materials", ["tenant_id"])

    # --- render_jobs -------------------------------------------------------
    op.create_table(
        "render_jobs",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("material_id", postgresql.UUID(as_uuid=True), nullable=True),
        # VARCHAR column per approved adjustment (not a native Postgres
        # enum type) — validated at the application layer via the
        # RenderJobStatus Python Enum.
        sa.Column("status", sa.String(length=20), nullable=False, server_default="pending"),
        sa.Column("input_image_url", sa.String(length=1024), nullable=False),
        sa.Column("output_image_url", sa.String(length=1024), nullable=True),
        sa.Column("error_message", sa.String(length=2048), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"], ["tenants.id"], name="fk_render_jobs_tenant_id_tenants", ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["material_id"],
            ["materials.id"],
            name="fk_render_jobs_material_id_materials",
            ondelete="SET NULL",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'completed', 'failed')",
            name="ck_render_jobs_status_valid",
        ),
    )
    op.create_index("ix_render_jobs_tenant_id", "render_jobs", ["tenant_id"])
    op.create_index("ix_render_jobs_material_id", "render_jobs", ["material_id"])

    # --- usage_records -------------------------------------------------------
    op.create_table(
        "usage_records",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("render_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("event_type", sa.String(length=100), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            name="fk_usage_records_tenant_id_tenants",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["render_job_id"],
            ["render_jobs.id"],
            name="fk_usage_records_render_job_id_render_jobs",
            ondelete="SET NULL",
        ),
    )
    op.create_index("ix_usage_records_tenant_id", "usage_records", ["tenant_id"])
    op.create_index("ix_usage_records_render_job_id", "usage_records", ["render_job_id"])


def downgrade() -> None:
    # Drop in reverse dependency order.
    op.drop_index("ix_usage_records_render_job_id", table_name="usage_records")
    op.drop_index("ix_usage_records_tenant_id", table_name="usage_records")
    op.drop_table("usage_records")

    op.drop_index("ix_render_jobs_material_id", table_name="render_jobs")
    op.drop_index("ix_render_jobs_tenant_id", table_name="render_jobs")
    op.drop_table("render_jobs")

    op.drop_index("ix_materials_tenant_id", table_name="materials")
    op.drop_table("materials")

    op.drop_index("ix_allowed_domains_tenant_id", table_name="allowed_domains")
    op.drop_table("allowed_domains")

    op.drop_index("ix_api_keys_key_hash", table_name="api_keys")
    op.drop_index("ix_api_keys_tenant_id", table_name="api_keys")
    op.drop_table("api_keys")

    op.drop_index("ix_tenants_slug", table_name="tenants")
    op.drop_table("tenants")

    # Not dropping the pgcrypto extension on downgrade — other parts of
    # the database may depend on it, and dropping extensions is rarely
    # desirable as part of an application migration rollback.
