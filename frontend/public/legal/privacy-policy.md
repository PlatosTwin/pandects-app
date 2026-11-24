# Pandects Privacy Policy

Effective date: **November 23, 2026**

**In short**
- We use cookies and similar storage for authentication, security, analytics, and certain site preferences.
- We use Google Analytics and limited client-side error telemetry, and analytics load by default when you visit the site.
- We use Google Identity Services for Google sign-in, Cloudflare Turnstile for signup abuse prevention when enabled, and Airtable for embedded feedback forms.
- We do not sell personal information.

---

Pandects is an open-source M&A research platform. We pull text from public filings on the SEC's EDGAR system, parse it, use AI to tag and classify it, and make resulting structured data available on this site and, in some cases, through APIs or downloadable datasets.

This Privacy Policy explains what information we collect when you use Pandects (the "Service"), how we use it, and the choices you may have.

If you have questions about this policy, contact us at **nmbogdan@alumni.stanford.edu**.

## 1. Who We Are

For purposes of this Privacy Policy, "Pandects," "we," "us," and "our" refer to the operators and maintainers of the Service. The Service may be operated by individuals or through one or more entities.

## 2. Scope

This Privacy Policy applies to information we collect through:

- The Pandects website and related web pages, including `/about`, `/terms`, `/privacy-policy`, `/license`, `/license/software`, and `/license/data`
- The Pandects API and related endpoints
- Account registration, login, password reset, Google sign-in, API key management, and related auth flows
- Feedback, survey, or contact forms embedded on the site, including third-party forms

This policy does not apply to third-party websites or services except as described here when they operate as our service providers on or for the Service. External sites linked from Pandects, such as GitHub, the SEC's EDGAR system, or other sites you visit separately, are governed by their own policies.

## 3. Information We Collect

### A. Information you provide

Depending on how you use the Service, you may provide:

- **Account information:** such as your email address. If password authentication is enabled, passwords are stored only as a cryptographic hash.
- **Google sign-in information:** if you use Google sign-in, we receive identity information sufficient to authenticate you, including your email address.
- **Feedback and survey responses:** information you submit via embedded forms, including Airtable forms. The content depends on the form.
- **Communications:** information you include when you email us or contact us through other channels.

You are responsible for ensuring that information you submit does not include sensitive personal information unless you intend to provide it.

### B. Information collected automatically

When you use the Service, we and our service providers may collect:

- **Access and server logs:** IP address, request timestamps, requested URLs, HTTP status codes, user agent strings, and referrer information, used to operate, secure, debug, and audit the Service.
- **Analytics data:** usage and interaction data collected via Google Analytics, including pageviews, outbound link clicks, time-on-page events, and similar usage metrics.
- **Client-side error telemetry:** limited browser-side error data, such as JavaScript error messages and file or line references, sent through Google Analytics.
- **Session and security data:** cookies or similar identifiers used for session management, Google auth nonce handling, OAuth state handling, authentication, and CSRF protection.
- **API usage data:** if you use an API key, we collect request counts, timestamps, key metadata, and related abuse-prevention or usage records. API keys are stored only as a cryptographic hash, along with a short prefix to help you identify a key.
- **Abuse-prevention and captcha data:** if Cloudflare Turnstile is enabled for signup or similar flows, Turnstile may collect information needed to assess whether a request is legitimate.
- **Browser storage data:** the frontend may use localStorage or sessionStorage for bearer-session auth mode, dismissing site banners, search display preferences, and cached UI data such as taxonomy or filter options.

### C. Cookies and similar technologies

Pandects uses cookies and similar technologies for:

- **Authentication and security:** maintaining sessions, login state, nonce or OAuth flow state, and protecting requests against forgery or abuse.
- **Analytics:** measuring usage and performance via Google Analytics.
- **Preferences and UX:** persisting certain site settings, such as sidebar state or display preferences, and caching some UI data in browser storage.

Analytics currently load by default when you visit the site; we do not currently present a separate analytics opt-in banner before loading Google Analytics.

You can control cookies and similar storage through your browser settings, but disabling them may affect site functionality.

### D. Service providers used on or for the Service

We currently use third-party providers such as:

- **Google Analytics:** for analytics and limited client-side error telemetry.
- **Google Identity Services:** for Google sign-in.
- **Cloudflare Turnstile:** for abuse prevention and captcha verification when enabled.
- **Airtable:** for embedded feedback forms.
- **Hosting and infrastructure providers:** to host, secure, and operate the site, APIs, and related data services.

These providers may process information according to their own terms and privacy policies, in addition to processing information on our behalf in connection with the Service.

### E. MCP server

When you connect an AI client, such as Claude, ChatGPT, Codex or Claude Code, to the Pandects MCP server at `https://api.pandects.org/mcp`, you sign in with your Pandects account and approve the client. The client then receives an access token for your account.

- **Tool usage records:** for each hour, we record which tools your account called, through which client registration, how many calls succeeded or failed, how long they took, and the total size of the requests. We use these records to operate the Service, prevent abuse, and enforce rate limits.
- **Tool arguments:** the arguments your client sends with a tool call, such as a search phrase, are used to answer that call. We do not store them in the usage records. Our server logs record the tool name, outcome and timing of each call, not its arguments. When a call fails with an unexpected error, the error log can hold part of the request. Our hosting provider keeps server logs for a limited time.
- **Conversations:** we do not receive or store your conversations with your AI client. We receive only the tool calls your client sends.
- **Feedback notes:** if you ask your AI client to send feedback with the `submit_feedback` tool, we store the note (summary, detail, suggestions, category, severity, the tool it is about, and a short context object) with the time it was sent, your account ID, the client registration and the token's scopes. Only the maintainer reads feedback notes. Do not include personal data in feedback.

We do not sell this information. We do not share it with the provider of your AI client, beyond the tool results your client requests. You can disconnect a client at any time in that client, which removes the client's copy of its tokens. If you delete your Pandects account, every connected client loses access at once.

## 4. How We Use Information

We use information we collect to:

- **Provide and operate** the Service, including authentication, API access, and dataset distribution.
- **Maintain security and integrity**, including preventing abuse, enforcing rate limits, verifying captcha challenges, and investigating suspicious activity.
- **Measure, analyze, and improve** the Service, including debugging, analytics, performance monitoring, and product improvements.
- **Communicate with you**, including responding to inquiries, feedback, support requests, and account-related notices.
- **Keep records of legal acceptance and account events**, including acceptance of legal documents and sign-in or registration events.

We do not use personal information to make automated decisions that produce legal or similarly significant effects.

## 5. How We Share Information

We do not sell personal information.

We may share information in the following circumstances:

- **Service providers:** with vendors that help us operate the Service, such as hosting providers, analytics providers, authentication providers, captcha providers, and form providers.
- **Legal and safety reasons:** to comply with applicable law, regulation, legal process, or governmental request, or to protect the rights, safety, and security of Pandects, our users, or the public.
- **Business changes:** if the Service is reorganized, transferred, or acquired, for example through a merger, acquisition, or asset sale, information may be transferred as part of that transaction.

## 6. Data Retention

We retain information for as long as reasonably necessary for the purposes described in this policy, including to operate the Service, comply with legal obligations, resolve disputes, and enforce agreements.

Examples:

- **Account information** is retained until you request deletion, subject to legal and operational requirements.
- **Legal acceptance and sign-in records** may be retained to document assent, account history, security events, and compliance.
- **API usage and security logs** may be retained to prevent abuse, enforce rate limits, and maintain system integrity.
- **MCP tool usage records and feedback notes** are retained until you ask us to delete them. If you delete your account, these records stay linked only to an account ID that no longer holds your email address, and you can ask us to remove them.
- **Analytics and telemetry data** are retained according to operational needs and the retention settings of our service providers.
- **Browser-side cached data** may remain on your device until cleared by you or overwritten by the application.

We may retain information in aggregated or de-identified form for longer periods.

## 7. Your Choices and Rights

Depending on your location and how you use the Service, you may have the right to:

- Request access to, correction of, or deletion of your account information.
- Request deactivation of your account and revocation of API keys.
- Disconnect an AI client from the MCP server in that client, and ask us to delete your MCP tool usage records and feedback notes.
- Clear cookies, localStorage, sessionStorage, or similar browser-side data from your device.
- Limit some analytics by blocking cookies or using browser or device controls.

To make a request, contact **nmbogdan@alumni.stanford.edu**. We may need to verify your identity before fulfilling certain requests, and some requests may be subject to legal or operational limitations.

## 8. Security

We take reasonable administrative, technical, and organizational measures to protect information. However, no system is completely secure. You are responsible for maintaining the confidentiality of your credentials and API keys.

## 9. Children and Minors

The Service is not intended for children or minors and is not available to anyone under 18 years of age. **If you are under 18, do not use the Service or provide personal information to us.**

We do not knowingly permit accounts for people under 18. If we learn that someone under 18 has used the Service or provided personal information, we may deactivate the account and delete or minimize the associated information as appropriate.

Separately, the Service is not directed to children under 13, and we do not knowingly collect personal information from children under 13.

## 10. Changes to This Policy

We may update this Privacy Policy from time to time. If we make material changes, we will update the effective date and, where appropriate, provide additional notice. Your continued use of the Service after changes become effective constitutes acceptance of the updated policy.

## 11. Contact

Pandects  
Email: **nmbogdan@alumni.stanford.edu**
