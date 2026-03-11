# ============================================================
# src/document_loader.py
# Load and parse DME billing documents from multiple formats
# Supports: PDF, Excel, CSV, Text, OCR fallback
# ============================================================

import os
import io
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
from pypdf import PdfReader
from PIL import Image

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


class DMEDocumentLoader:
    """
    Loads DME billing documents from PDF, Excel, CSV, and text files.
    Each document is split into overlapping chunks for vector storage.

    Why chunking? LLMs have a context window limit. We split large
    documents into smaller pieces so each piece fits. Overlap ensures
    we don't lose context at chunk boundaries.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # RecursiveCharacterTextSplitter tries to split on natural
        # boundaries: paragraphs → sentences → words → characters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    # ── PDF ──────────────────────────────────────────────────

    def load_pdf(self, file_path: str) -> List[Document]:
        """Extract text from PDF files (billing forms, CMNs, EOBs)."""
        documents = []
        try:
            reader = PdfReader(file_path)
        except Exception as e:
            logger.error(f"Failed to open PDF {file_path}: {e}")
            return []

        for page_num, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if len(text.strip()) < 20:
                # Scanned page — attempt OCR
                logger.info(f"Page {page_num + 1} is scanned, attempting OCR...")
                text = self._ocr_page_via_pymupdf(file_path, page_num)

            if text and len(text.strip()) > 10:
                documents.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": file_path,
                            "page": page_num + 1,
                            "file_type": "pdf",
                            "filename": Path(file_path).name,
                        },
                    )
                )

        logger.info(f"Loaded PDF: {Path(file_path).name} — {len(documents)} pages")
        return documents

    # ── Excel ────────────────────────────────────────────────

    def load_excel(self, file_path: str) -> List[Document]:
        """Convert Excel spreadsheets to text (billing codes, fee schedules)."""
        documents = []
        try:
            excel_file = pd.ExcelFile(file_path)
        except Exception as e:
            logger.error(f"Failed to open Excel {file_path}: {e}")
            return []

        for sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                df = df.dropna(how="all")

                lines = [
                    f"File: {Path(file_path).name}",
                    f"Sheet: {sheet_name}",
                    f"Columns: {', '.join(str(c) for c in df.columns.tolist())}",
                    f"Rows: {len(df)}",
                    "",
                ]

                for _, row in df.iterrows():
                    row_text = " | ".join(
                        f"{col}: {val}"
                        for col, val in row.items()
                        if pd.notna(val) and str(val).strip()
                    )
                    if row_text:
                        lines.append(row_text)

                documents.append(
                    Document(
                        page_content="\n".join(lines),
                        metadata={
                            "source": file_path,
                            "sheet": sheet_name,
                            "file_type": "excel",
                            "filename": Path(file_path).name,
                            "rows": len(df),
                        },
                    )
                )
            except Exception as e:
                logger.warning(f"Could not read sheet '{sheet_name}': {e}")

        logger.info(f"Loaded Excel: {Path(file_path).name} — {len(documents)} sheets")
        return documents

    # ── CSV ──────────────────────────────────────────────────

    def load_csv(self, file_path: str) -> List[Document]:
        """Load CSV files (claim exports, fee schedules)."""
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logger.error(f"Failed to open CSV {file_path}: {e}")
            return []

        lines = [
            f"File: {Path(file_path).name}",
            f"Columns: {', '.join(df.columns.tolist())}",
            f"Total rows: {len(df)}",
            "",
        ]

        for _, row in df.iterrows():
            row_text = " | ".join(
                f"{col}: {val}" for col, val in row.items() if pd.notna(val)
            )
            if row_text:
                lines.append(row_text)

        doc = Document(
            page_content="\n".join(lines),
            metadata={
                "source": file_path,
                "file_type": "csv",
                "filename": Path(file_path).name,
            },
        )

        logger.info(f"Loaded CSV: {Path(file_path).name}")
        return [doc]

    # ── Plain Text ───────────────────────────────────────────

    def load_text(self, file_path: str) -> List[Document]:
        """Load plain text / markdown files (policies, guidelines)."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except Exception as e:
            logger.error(f"Failed to open text file {file_path}: {e}")
            return []

        doc = Document(
            page_content=content,
            metadata={
                "source": file_path,
                "file_type": "text",
                "filename": Path(file_path).name,
            },
        )

        logger.info(f"Loaded text: {Path(file_path).name}")
        return [doc]

    # ── Directory ────────────────────────────────────────────

    def load_directory(self, directory_path: str) -> List[Document]:
        """Recursively load all supported documents from a directory."""
        all_documents: List[Document] = []
        directory = Path(directory_path)

        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory_path}")
            return []

        extension_map = {
            ".pdf": self.load_pdf,
            ".xlsx": self.load_excel,
            ".xls": self.load_excel,
            ".csv": self.load_csv,
            ".txt": self.load_text,
            ".md": self.load_text,
        }

        for file_path in sorted(directory.rglob("*")):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in extension_map:
                    logger.info(f"Loading: {file_path.name}")
                    docs = extension_map[ext](str(file_path))
                    all_documents.extend(docs)

        logger.info(f"Total raw documents loaded: {len(all_documents)}")
        return all_documents

    # ── Chunking ─────────────────────────────────────────────

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into overlapping chunks ready for embedding."""
        if not documents:
            return []
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks

    # ── OCR Fallback ─────────────────────────────────────────

    def _ocr_page_via_pymupdf(self, pdf_path: str, page_num: int) -> str:
        """Use PyMuPDF + pytesseract to OCR a scanned PDF page."""
        try:
            import fitz  # PyMuPDF
            import pytesseract

            doc = fitz.open(pdf_path)
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img)
            doc.close()
            return text
        except ImportError:
            logger.warning("PyMuPDF or pytesseract not available for OCR")
            return ""
        except Exception as e:
            logger.warning(f"OCR failed on page {page_num + 1}: {e}")
            return ""
