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

const SECTION_HEADERS = [
  'OVERVIEW',
  'VISIBLE FEATURES',
  'SPATIAL PATTERN',
  'INTERPRETATION',
  'VISUAL OBSERVATIONS',
  'SUMMARY',
  'KEY FINDINGS',
  'LAND COVER ANALYSIS',
  'OBJECT LOCALIZATION'
];

const BOLD_SPLIT_REGEX = /(\*\*[^*]+\*\*)/g;

function isHeaderLine(line: string): { isHeader: boolean; heading: string } {
  const trimmed = line.trim();
  if (!trimmed) return { isHeader: false, heading: '' };

  // Strip leading markdown hashes, bolding, numbering, and trailing colons
  let cleanLine = trimmed
    .replace(/^#{1,6}\s*/, '')
    .replace(/^\*\*|\*\*$/g, '')
    .replace(/^__|_$/g, '')
    .replace(/^\d+[\.\)]\s*/, '')
    .replace(/:$/, '')
    .trim();

  // Second pass in case of nested wrapping like **1. OVERVIEW**
  cleanLine = cleanLine
    .replace(/^#{1,6}\s*/, '')
    .replace(/^\*\*|\*\*$/g, '')
    .replace(/^\d+[\.\)]\s*/, '')
    .replace(/:$/, '')
    .trim();

  // Check against known section headers (case-insensitive)
  for (const h of SECTION_HEADERS) {
    if (cleanLine.toUpperCase() === h) {
      return { isHeader: true, heading: h };
    }
  }

  // Check if all-caps short heading (e.g. "GEOSPATIAL CONTEXT")
  if (/^[A-Z][A-Z0-9 /&()_-]{2,30}$/.test(cleanLine) && !cleanLine.includes('.')) {
    return { isHeader: true, heading: cleanLine.toUpperCase() };
  }

  return { isHeader: false, heading: '' };
}

function parseAnswer(raw: string): AnswerBlock[] {
  const text = raw.trim();
  if (!text) return [];

  const lines = text.split('\n');
  const blocks: AnswerBlock[] = [];

  let currentHeading: string | null = null;
  let currentLines: string[] = [];

  const flushCurrent = () => {
    if (currentHeading) {
      blocks.push({ type: 'heading', heading: currentHeading });
      currentHeading = null;
    }

    if (currentLines.length === 0) return;

    // Check if lines are bullet points or numbered list
    const hasBullets = currentLines.some((l) => /^[-*•]\s+/.test(l.trim()));
    const hasNumbered = currentLines.some((l) => /^\d+\.\s+/.test(l.trim()));

    if (hasBullets) {
      const items: string[] = [];
      let currentItem = '';

      for (const line of currentLines) {
        const tr = line.trim();
        if (/^[-*•]\s+/.test(tr)) {
          if (currentItem) items.push(currentItem);
          currentItem = tr.replace(/^[-*•]\s+/, '').trim();
        } else if (tr) {
          if (currentItem) {
            currentItem += ' ' + tr;
          } else {
            currentItem = tr;
          }
        }
      }
      if (currentItem) items.push(currentItem);

      if (items.length > 0) {
        blocks.push({ type: 'bullets', items });
      }
    } else if (hasNumbered) {
      const items: NumberedItem[] = [];
      let currentTitle = '';
      let subItems: string[] = [];

      for (const line of currentLines) {
        const tr = line.trim();
        if (/^\d+\.\s+/.test(tr)) {
          if (currentTitle) {
            items.push({ title: currentTitle, subItems });
            subItems = [];
          }
          currentTitle = tr.replace(/^\d+\.\s+/, '').trim();
        } else if (/^[-*•]\s+/.test(tr)) {
          subItems.push(tr.replace(/^[-*•]\s+/, '').trim());
        } else if (tr) {
          if (currentTitle) {
            currentTitle += ' ' + tr;
          } else {
            currentTitle = tr;
          }
        }
      }
      if (currentTitle) items.push({ title: currentTitle, subItems });

      if (items.length > 0) {
        blocks.push({ type: 'numbered', numberedItems: items });
      }
    } else {
      const para = currentLines.join(' ').trim();
      if (para) {
        blocks.push({ type: 'paragraph', text: para });
      }
    }

    currentLines = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const { isHeader, heading } = isHeaderLine(line);

    if (isHeader) {
      flushCurrent();
      currentHeading = heading;
    } else if (line.trim() === '') {
      if (currentLines.length > 0 && !currentLines.some((l) => /^[-*•\d.]+\s+/.test(l.trim()))) {
        flushCurrent();
      }
    } else {
      currentLines.push(line);
    }
  }

  flushCurrent();

  return blocks;
}

function renderInline(text: string, keyPrefix: string): React.ReactNode {
  // If the text starts with "Title: " without markdown bolding, format the label as bold accent
  let processedText = text;
  if (!text.startsWith('**') && /^([A-Za-z0-9 /&()_-]+):\s+(.*)$/.test(text)) {
    const match = text.match(/^([A-Za-z0-9 /&()_-]+):\s+(.*)$/);
    if (match) {
      processedText = `**${match[1]}**: ${match[2]}`;
    }
  }

  const parts = processedText.split(BOLD_SPLIT_REGEX).filter((p) => p.length > 0);
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
    <div className="space-y-2.5">
      {blocks.map((block, idx) => {
        if (block.type === 'heading') {
          return (
            <h4
              key={idx}
              className="text-[11px] font-mono font-bold text-sat-accent uppercase tracking-wider pt-2 first:pt-0"
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
                  <span className="mt-[5px] w-1.5 h-1.5 rounded-full bg-sat-cyan flex-shrink-0" />
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
                          <span className="mt-[5px] w-1.5 h-1.5 rounded-full bg-sat-cyan/70 flex-shrink-0" />
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
