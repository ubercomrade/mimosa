"""Compatibility facade for motif file readers and writers."""

from __future__ import annotations

from mimosa.io.bamm import parse_file_content, read_bamm
from mimosa.io.dist import write_dist
from mimosa.io.meme import read_meme, read_meme_many
from mimosa.io.pfm import read_pfm, write_pfm
from mimosa.io.sitega import read_sitega, write_sitega

__all__ = [
    "parse_file_content",
    "read_bamm",
    "read_meme",
    "read_meme_many",
    "read_pfm",
    "read_sitega",
    "write_dist",
    "write_pfm",
    "write_sitega",
]
