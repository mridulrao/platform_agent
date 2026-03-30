import logging
from datetime import datetime
from typing import Any, Dict, List

from sync_kb_helper import (
    extract_kb_metadata,
    generate_article_summary,
    get_openai_embeddings_batch,
)

from kb_utils.kb_db import call_with_429_retries


logger = logging.getLogger(__name__)


class KnowledgeBaseArticleProcessor:
    def __init__(self, *, kb_collection_id: str):
        self.kb_collection_id = kb_collection_id

    def process_single_article(
        self,
        kbid: str,
        kb_title: str,
        kb_content: str,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {"kbid": kbid, "success": False, "error": None, "stage": None}

        try:
            result["stage"] = "metadata_extraction"
            metadata = extract_kb_metadata(kb_content=kb_content, kbid=kbid, kb_title=kb_title)

            result["stage"] = "summary_generation"
            summary_obj = call_with_429_retries(
                generate_article_summary,
                document=kb_content,
                document_title=kb_title,
                kbid=kbid,
            )

            category = metadata.get("category", "")
            article_type = metadata.get("article_type", "")
            primary_topic = metadata.get("primary_topic", "")

            kw = metadata.get("keywords", {}) or {}
            primary_specific = kw.get("primary_specific", []) or []
            primary_generic = kw.get("primary_generic", []) or []
            secondary_entities = kw.get("secondary_entities", []) or []
            acronyms = kw.get("acronyms", []) or []
            usecases = metadata.get("usecase", []) or []

            if not primary_specific and not primary_generic and not usecases:
                raise ValueError("No primary_specific/primary_generic/usecase extracted")

            result["stage"] = "embedding_generation"
            texts: List[str] = (
                list(primary_specific)
                + list(primary_generic)
                + list(usecases)
                + list(secondary_entities)
                + list(acronyms)
            )
            embeddings = call_with_429_retries(get_openai_embeddings_batch, texts) if texts else []
            if len(embeddings) != len(texts):
                raise ValueError(
                    f"Embedding count mismatch: expected {len(texts)}, got {len(embeddings)}"
                )

            result["stage"] = "row_preparation"
            now = datetime.now()
            keyword_rows: List[Dict[str, Any]] = []
            emb_i = 0

            def next_emb() -> Any:
                nonlocal emb_i
                value = embeddings[emb_i]
                emb_i += 1
                return value

            base_row = {
                "kbid": kbid,
                "kb_collection_id": self.kb_collection_id,
                "category": category,
                "article_type": article_type,
                "primary_topic": primary_topic,
                "specific_keyword": None,
                "generic_keyword": None,
                "usecase": None,
                "secondary_entity": None,
                "acronym": None,
                "specific_keyword_embedding": None,
                "generic_keyword_embedding": None,
                "usecase_embedding": None,
                "secondary_entity_embedding": None,
                "acronym_embedding": None,
                "created_at": now,
                "updated_at": now,
            }

            for keyword in primary_specific:
                keyword_rows.append({**base_row, "specific_keyword": keyword, "specific_keyword_embedding": next_emb()})
            for keyword in primary_generic:
                keyword_rows.append({**base_row, "generic_keyword": keyword, "generic_keyword_embedding": next_emb()})
            for usecase in usecases:
                keyword_rows.append({**base_row, "usecase": usecase, "usecase_embedding": next_emb()})
            for entity in secondary_entities:
                keyword_rows.append({**base_row, "secondary_entity": entity, "secondary_entity_embedding": next_emb()})
            for acronym in acronyms:
                keyword_rows.append({**base_row, "acronym": acronym, "acronym_embedding": next_emb()})

            query_triggers = summary_obj.get("query_triggers", []) or []
            secondary_mentions = summary_obj.get("secondary_mentions", []) or []

            summary_data = {
                "doc_id": kbid,
                "kb_collection_id": self.kb_collection_id,
                "article_title": (summary_obj.get("article_title") or kb_title or "").strip(),
                "primary_intent": (summary_obj.get("primary_intent") or "").strip(),
                "query_triggers": ", ".join(str(x) for x in query_triggers if str(x).strip()),
                "secondary_mentions": ", ".join(
                    str(x) for x in secondary_mentions if str(x).strip()
                ),
                "summary": (summary_obj.get("summary") or "").strip(),
                "content": kb_content,
                "created_at": now,
                "updated_at": now,
            }

            result["keyword_rows"] = keyword_rows
            result["summary_data"] = summary_data
            result["metadata"] = {
                "keyword_rows": len(keyword_rows),
                "primary_specific": len(primary_specific),
                "primary_generic": len(primary_generic),
                "usecases": len(usecases),
                "secondary_entities": len(secondary_entities),
                "acronyms": len(acronyms),
                "category": category,
                "article_type": article_type,
            }
            result["success"] = True
            result["stage"] = "completed"
            return result
        except Exception as exc:
            result["error"] = str(exc)
            result["success"] = False
            return result
