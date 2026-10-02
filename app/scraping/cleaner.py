import re

from app.schemas import SourceDocument


class BankContentCleaner:
    BOILERPLATE_EXACT_LINES = {
        "${loading}",
        "${title}",
        "{}",
        "abrir cuenta",
        "aceptar",
        "activación",
        "activar tarjeta",
        "angle-right-small",
        "aprende más",
        "arrow2-down",
        "arrow2-right",
        "card",
        "card-contactless",
        "cards",
        "circle-question",
        "conoce más",
        "crear tarjeta",
        "error",
        "flight",
        "globe",
        "hand-holding-cash",
        "inicio",
        "investment",
        "key",
        "laptop",
        "menú",
        "menu-dots-v",
        "money",
        "pocket",
        "plus",
        "plant",
        "puntos-colombia",
        "send-money",
        "send-money-to",
        "shield-money",
        "star",
        "tag-price",
        "trophy",
        "users",
    }

    BOILERPLATE_PREFIXES = (
        "aceptar cookies",
        "bancolombia personas {}",
        "configurar cookies",
        "educación financiera",
        "educacion financiera",
        "menú de acciones de componente",
        "mapa del sitio",
        "mostrar menú de portlet",
        "mostrar menu de portlet",
        "líneas de atención",
        "lineas de atencion",
        "personas negocios",
        "productos y servicios",
        "síguenos",
        "siguenos",
        "sucursal virtual personas",
        "trámites digitales",
        "tramites digitales",
        "transparencia consumidor",
        "visor de contenido web",
    )

    def clean(self, documents: list[SourceDocument]) -> list[SourceDocument]:
        cleaned_documents: list[SourceDocument] = []

        for document in documents:
            cleaned_text = self.clean_text(document.content)
            if not cleaned_text:
                continue

            cleaned_documents.append(document.model_copy(update={"content": cleaned_text}))

        return cleaned_documents

    def clean_text(self, text: str) -> str:
        normalized = self._normalize_whitespace(text)
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        lines = self._remove_global_header(lines)
        lines = self._remove_global_footer(lines)
        lines = self._remove_boilerplate_lines(lines)
        lines = self._deduplicate_consecutive_lines(lines)
        lines = self._deduplicate_repeated_lines(lines)
        return "\n".join(lines).strip()

    def _remove_global_header(self, lines: list[str]) -> list[str]:
        if not lines or self._normalize_for_comparison(lines[0]) != "{}":
            return lines

        navigation_end_markers = {
            "bam (banco agromercantil de guatemala)",
            "banco agromercantil de guatemala)",
        }
        navigation_end_index: int | None = None
        for index, line in enumerate(lines[:220]):
            if self._normalize_for_comparison(line) in navigation_end_markers:
                navigation_end_index = index + 1
            if self._is_split_bam_marker(lines, index):
                navigation_end_index = index + 4

        if navigation_end_index is not None:
            return lines[navigation_end_index:]
        return lines

    def _is_split_bam_marker(self, lines: list[str], index: int) -> bool:
        if index + 3 >= len(lines):
            return False
        window = [self._normalize_for_comparison(line) for line in lines[index : index + 4]]
        return window == ["bam", "(banco agromercantil", "de", "guatemala)"]

    def _remove_global_footer(self, lines: list[str]) -> list[str]:
        footer_markers = {
            "te puede interesar",
            "accesibilidad",
            "notificaciones judiciales",
            "contáctanos",
            "contactanos",
        }
        copyright_index = next(
            (
                index
                for index, line in enumerate(lines)
                if "copyright" in self._normalize_for_comparison(line)
            ),
            None,
        )
        if copyright_index is None:
            return lines

        for index, line in enumerate(lines[:copyright_index]):
            if self._normalize_for_comparison(line) in footer_markers:
                return lines[:index]
        return lines

    def _normalize_whitespace(self, text: str) -> str:
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def _remove_boilerplate_lines(self, lines: list[str]) -> list[str]:
        filtered: list[str] = []
        for line in lines:
            normalized = self._normalize_for_comparison(line)
            if normalized in self.BOILERPLATE_EXACT_LINES:
                continue
            if any(normalized.startswith(prefix) for prefix in self.BOILERPLATE_PREFIXES):
                continue
            if self._looks_like_icon_token(normalized):
                continue
            filtered.append(line)
        return filtered

    def _deduplicate_consecutive_lines(self, lines: list[str]) -> list[str]:
        deduplicated: list[str] = []
        previous = ""

        for line in lines:
            comparable = self._normalize_for_comparison(line)
            if comparable == previous:
                continue
            deduplicated.append(line)
            previous = comparable

        return deduplicated

    def _deduplicate_repeated_lines(self, lines: list[str]) -> list[str]:
        deduplicated: list[str] = []
        seen: set[str] = set()

        for line in lines:
            comparable = self._normalize_for_comparison(line)
            if comparable in seen:
                continue
            deduplicated.append(line)
            seen.add(comparable)

        return deduplicated

    def _looks_like_icon_token(self, text: str) -> bool:
        return bool(re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)+", text)) and len(text) <= 40

    def _normalize_for_comparison(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.casefold()).strip()
