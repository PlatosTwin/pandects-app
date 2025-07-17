import Navigation from "@/components/Navigation";

export default function Docs() {
  return (
    <div className="min-h-screen bg-cream flex flex-col">
      <Navigation />
      <main className="flex-1 p-12">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-material-text-primary mb-8">
            API Documentation
          </h1>

          <div className="bg-white rounded-lg border border-gray-200 p-8 shadow-sm">
            <p className="text-material-text-secondary text-lg">
              API documentation is coming soon. This page will contain
              comprehensive guides and references for using the Pandects API.
            </p>
          </div>
        </div>
      </main>
    </div>
  );
}
