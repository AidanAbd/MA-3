import allTags from 'html-tags';
import singletonTags from 'html-tags/void';
import legalAttrs from 'html-element-attributes';

import htmlEscape from 'escape-html';

const globalLegalAttrs = legalAttrs['*'];

export const dom = (tag, attrs, ...children) => {
  if (!allTags.includes(tag))
    throw new Error(`Unknown HTML tag ${tag}.`);

  const res = document.createElement(tag);

  const curLegalAttrs = legalAttrs[tag];
  if (attrs != null) {
    for (const [k, v] of Object.entries(attrs)) {
      if (!globalLegalAttrs.includes(k) &&
          curLegalAttrs != null &&
          !curLegalAttrs.includes(k) &&
          !k.startsWith('data-'))
        throw new Error(`Attribute ${k} illegal for tag ${tag}.`);

      if (k !== htmlEscape(k))
        throw new Error(`Attribute ${k} contains illegal characters.`);

      if (v === true)
        res.setAttribute(k, '');
      else
        res.setAttribute(k, htmlEscape(v));
    }
  }

  if (singletonTags.includes(tag)) {
    if (children.length !== 0)
      throw new Error(`Have a child in a singleton tag ${tag}.`);
  }

  for (const c of children) {
    if (c instanceof Element) {
      res.appendChild(c);
      continue;
    }
    if (typeof c === 'object') {
      if (Object.keys(c).length !== 1 || c.el == null) {
        throw new Error(`Child ${JSON.stringify(c)} is a weird object. Has keys: ${Object.keys(c)}, but must only have an \`el\` attribute.`);
      }
      res.appendChild(c.el);
      continue;
    }

    if (typeof c !== 'string')
      throw new Error(`Child ${c} is not a string.`);

    res.appendChild(document.createTextNode(htmlEscape(c)));
  }

  return {el: res};
};
