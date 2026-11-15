# AcmeCloud Platform Documentation

## Authentication & SSO
AcmeCloud supports SAML 2.0 and OIDC for Enterprise Single Sign-On (SSO).
To enable SSO, navigate to **Settings > Security > SSO** and upload your Identity Provider (IdP) metadata XML.
Session tokens expire after 12 hours by default and can be configured down to 1 hour for high-security workspaces.

## API Rate Limits
- **Free Tier:** 60 requests per minute per API key.
- **Pro Tier:** 600 requests per minute per API key.
- **Enterprise Tier:** Custom burst limits up to 5,000 requests per minute.
When a client exceeds the rate limit, the API returns HTTP `429 Too Many Requests` with a `Retry-After` header.

## Webhooks
Webhooks deliver real-time event notifications to your HTTPS endpoint.
Every webhook payload is signed using HMAC-SHA256 in the `X-Acme-Signature` header.
If your endpoint fails to respond with a `2xx` status code within 5 seconds, AcmeCloud retries with exponential backoff up to 5 times over 24 hours.
