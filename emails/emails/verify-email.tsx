import {
  Body,
  Button,
  Container,
  Head,
  Heading,
  Hr,
  Html,
  Link,
  Preview,
  Section,
  Text,
} from '@react-email/components';

export interface VerifyEmailProps {
  VERIFY_URL: string;
  YEAR: string;
}

export const VerifyEmail = ({ VERIFY_URL, YEAR }: VerifyEmailProps) => {
  const previewText = 'Verify your email address';

  return (
    <Html>
      <Head />
      <Preview>{previewText}</Preview>
      <Body style={main}>
        <Container style={container}>
          <Section style={brandSection}>
            <Text style={brandText}>Pandects</Text>
          </Section>

          <Heading style={title}>Verify your email</Heading>
          <Text style={copy}>
            Thanks for signing up. Verify your email to gain access to the API and unlock full search results.
          </Text>

          <Section style={buttonSection}>
            <Button href={VERIFY_URL} style={button}>
              Verify email
            </Button>
          </Section>

          <Text style={note}>
            If the button doesn&apos;t work, copy and paste this URL into your
            browser:
          </Text>
          <Section style={linkSection}>
            <Link href={VERIFY_URL} style={link}>
              {VERIFY_URL}
            </Link>
          </Section>

          <Hr style={divider} />

          <Text style={note}>
            If you didn&apos;t request this email, you can safely ignore it.
            Questions? Email{' '}
            <Link
              href="mailto:nmbogdan@alumni.stanford.edu"
              style={emailLink}
            >
              nmbogdan@alumni.stanford.edu
            </Link>
            .
          </Text>

          <Text style={footer}>© {YEAR} Pandects</Text>
        </Container>
      </Body>
    </Html>
  );
};

VerifyEmail.PreviewProps = {
  VERIFY_URL: '{{{VERIFY_URL}}}',
  YEAR: '{{{YEAR}}}',
} as VerifyEmailProps;

export default VerifyEmail;

const main = {
  backgroundColor: '#fbfaf5',
  fontFamily:
    'ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI","Helvetica Neue",Arial,sans-serif',
  padding: '12px',
};

const container = {
  backgroundColor: '#ffffff',
  border: '1px solid #e2e8f0',
  margin: '44px auto',
  maxWidth: '520px',
  padding: '32px',
};

const brandSection = {
  marginBottom: '20px',
};

const brandText = {
  color: '#0f172a',
  fontSize: '14px',
  fontWeight: 600,
  margin: '0',
};

const title = {
  color: '#0f172a',
  fontSize: '22px',
  fontWeight: 600,
  letterSpacing: '-0.01em',
  margin: '0 0 10px',
};

const copy = {
  color: '#334155',
  fontSize: '14px',
  lineHeight: '22px',
  margin: '0',
};

const buttonSection = {
  margin: '24px 0',
  textAlign: 'center' as const,
};

const button = {
  backgroundColor: '#2563eb',
  color: '#ffffff',
  display: 'inline-block',
  fontSize: '14px',
  fontWeight: 600,
  padding: '12px 20px',
  textDecoration: 'none',
};

const note = {
  color: '#64748b',
  fontSize: '12px',
  lineHeight: '18px',
  margin: '0',
};

const linkSection = {
  backgroundColor: '#f8fafc',
  border: '1px solid #e2e8f0',
  marginTop: '10px',
  padding: '10px 12px',
};

const link = {
  color: '#0f172a',
  display: 'block',
  fontFamily: 'ui-monospace,SFMono-Regular,Menlo,Monaco,monospace',
  fontSize: '12px',
  lineHeight: '18px',
  textDecoration: 'none',
  wordBreak: 'break-all' as const,
};

const emailLink = {
  color: '#64748b',
  fontSize: '12px',
  lineHeight: '18px',
  textDecoration: 'underline',
};

const divider = {
  borderColor: '#e2e8f0',
  margin: '24px 0',
};

const footer = {
  color: '#64748b',
  fontSize: '12px',
  lineHeight: '18px',
  margin: '16px 0 0',
};
