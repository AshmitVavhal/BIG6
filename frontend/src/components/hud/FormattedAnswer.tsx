import React from 'react';

interface NumberedItem {
  title: string;
  subItems: string[];
}

interface AnswerBlock {
  type: 'heading' | 'paragraph' | 'bullets' | 'numbered';
  heading?: string;
  text?: string;
  items?: string[];
  numberedItems?: NumberedItem[];
}

// Matches short capitalized labels like "Overview:", "Spatial Pattern:", or markdown
// "### Summary:" / "### Spatial Distribution" headers that start a new sentence —
// terminated either by a colon+space, or (for "###" headers with no colon) right
// before a bullet marker. Never matches bullet labels such as "**Residential Housing:**",
// since there the colon is immediately followed by "**" rather than whitespace.
const HEADER_REGEX = /(?:^|(?<=[.!?:]\s))(?:#{1,6}\s+)?([A-Z][a-zA-Z]+(?:[ /&][A-Z]?[a-zA-Z()0-9&]+){0,4})(?=:(?=\s)|\s[-*]\s\*\*)/g;
const BULLET_SPLIT_REGEX = /\s[-*]\s(?=\*\*)/g;
const HAS_BULLETS_REGEX = /[-*]\s\*\*/;
const NUMBERED_MARKER_REGEX = /(?:^|\s)(\d+)\.\s(?=\*\*)/g;
const HAS_NUMBERED_REGEX = /\d+\.\s\*\*/;
const BOLD_SPLIT_REGEX = /(\*\*[^*]+\*\*)/g;

function splitNumbered(content: string): string[] {
  const positions: { index: number; length: number }[] = [];
  let mm: RegExpExecArray | null;
  NUMBERED_MARKER_REGEX.lastIndex = 0;
  while ((mm = NUMBERED_MARKER_REGEX.exec(content)) !== null) {
    positions.push({ index: mm.index, length: mm[0].length });
  }
  const items: string[] = [];
  for (let i = 0; i < positions.length; i++) {
    const start = positions[i].index + positions[i].length;
    const end = i + 1 < positions.length ? positions[i + 1].index : content.length;
    items.push(content.slice(start, end).trim());
  }
  return items;
}

function parseAnswer(raw: string): AnswerBlock[] {
  const text = raw.trim();
  if (!text) return [];

  const matches: { index: number; end: number; heading: string }[] = [];
  let m: RegExpExecArray | null;
  HEADER_REGEX.lastIndex = 0;
  while ((m = HEADER_REGEX.exec(text)) !== null) {
    let end = m.index + m[0].length;
    if (text[end] === ':') end += 1;
    matches.push({ index: m.index, end, heading: m[1] });
  }

  const blocks: AnswerBlock[] = [];

  const pushContent = (heading: string | null, content: string) => {
    if (!content.trim()) return;
    if (heading) blocks.push({ type: 'heading', heading });

    if (HAS_NUMBERED_REGEX.test(content)) {
      const numberedItems = splitNumbered(content).map((raw) => {
        const parts = raw.split(BULLET_SPLIT_REGEX).map((s) => s.trim()).filter(Boolean);
        return { title: parts[0] ?? '', subItems: parts.slice(1) };
      });
      blocks.push({ type: 'numbered', numberedItems });
    } else if (HAS_BULLETS_REGEX.test(content)) {
      const items = content
        .split(BULLET_SPLIT_REGEX)
        .map((s) => s.trim())
        .filter(Boolean);
      if (items.length > 0) blocks.push({ type: 'bullets', items });
    } else {
      blocks.push({ type: 'paragraph', text: content.trim() });
    }
  };

  if (matches.length === 0) {
    pushContent(null, text);
    return blocks;
  }

  if (matches[0].index > 0) {
    pushContent(null, text.slice(0, matches[0].index));
  }

  for (let i = 0; i < matches.length; i++) {
    const start = matches[i].end;
    const end = i + 1 < matches.length ? matches[i + 1].index : text.length;
    pushContent(matches[i].heading, text.slice(start, end));
  }

  return blocks;
}

function renderInline(text: string, keyPrefix: string): React.ReactNode {
  const parts = text.split(BOLD_SPLIT_REGEX).filter((p) => p.length > 0);
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return (
        <strong key={`${keyPrefix}-b-${i}`} className="text-sat-accent font-semibold">
          {part.slice(2, -2)}
        </strong>
      );
    }
    return <React.Fragment key={`${keyPrefix}-t-${i}`}>{part}</React.Fragment>;
  });
}

interface FormattedAnswerProps {
  text: string;
}

export const FormattedAnswer: React.FC<FormattedAnswerProps> = ({ text }) => {
  const blocks = React.useMemo(() => parseAnswer(text), [text]);

  if (blocks.length === 0) return null;

  return (
    <div className="space-y-2">
      {blocks.map((block, idx) => {
        if (block.type === 'heading') {
          return (
            <h4
              key={idx}
              className="text-[11px] font-mono font-bold text-sat-accent uppercase tracking-wide pt-1 first:pt-0"
            >
              {block.heading}
            </h4>
          );
        }
        if (block.type === 'bullets') {
          return (
            <ul key={idx} className="space-y-1.5 pl-0.5">
              {block.items!.map((item, i) => (
                <li key={i} className="flex items-start space-x-2 text-xs text-sat-text leading-relaxed">
                  <span className="mt-[5px] w-1 h-1 rounded-full bg-sat-cyan flex-shrink-0" />
                  <span>{renderInline(item, `${idx}-${i}`)}</span>
                </li>
              ))}
            </ul>
          );
        }
        if (block.type === 'numbered') {
          return (
            <ol key={idx} className="space-y-2.5 pl-0.5">
              {block.numberedItems!.map((item, i) => (
                <li key={i} className="text-xs text-sat-text leading-relaxed">
                  <div className="flex items-start space-x-2">
                    <span className="font-mono text-[10px] font-bold text-sat-cyan mt-0.5 flex-shrink-0">
                      {i + 1}.
                    </span>
                    <span>{renderInline(item.title, `${idx}-${i}-title`)}</span>
                  </div>
                  {item.subItems.length > 0 && (
                    <ul className="mt-1.5 ml-4 space-y-1.5 border-l border-sat-border pl-3">
                      {item.subItems.map((sub, j) => (
                        <li key={j} className="flex items-start space-x-2">
                          <span className="mt-[5px] w-1 h-1 rounded-full bg-sat-cyan/70 flex-shrink-0" />
                          <span>{renderInline(sub, `${idx}-${i}-${j}`)}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </li>
              ))}
            </ol>
          );
        }
        return (
          <p key={idx} className="text-xs text-sat-text leading-relaxed">
            {renderInline(block.text!, `${idx}`)}
          </p>
        );
      })}
    </div>
  );
};
