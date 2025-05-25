import { Link } from "react-router-dom";

function HomePage() {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <h1 className="text-3xl font-extrabold text-center text-transparent bg-clip-text bg-gradient-to-r from-sky-500 to-blue-800 mb-10 tracking-wide leading-tight drop-shadow-md">
        Image Similarity Detection for CTU Publications
      </h1>

      <div className="flex flex-wrap justify-center gap-8">
        <Link to="/similarity" className="block">
          <div className="w-64 h-64 rounded-2xl shadow-md hover:shadow-xl transition duration-300 transform hover:-translate-y-1 bg-gradient-to-br from-sky-400 to-blue-500">
            <div className="w-full h-full flex flex-col items-center justify-center p-6 text-white">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-14 w-14 mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
              <span className="text-lg font-semibold tracking-wide">
                Image Similarity
              </span>
              <p className="text-sm text-center mt-2 leading-snug opacity-90">
                Compare a single image with images from Can Tho University
                Science Journal and Publishing House.{" "}
              </p>
            </div>
          </div>
        </Link>

        <Link to="/pdf-similarity" className="block">
          <div className="w-64 h-64 rounded-2xl shadow-md hover:shadow-xl transition duration-300 transform hover:-translate-y-1 bg-gradient-to-br from-emerald-400 to-teal-500">
            <div className="w-full h-full flex flex-col items-center justify-center p-6 text-white">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-14 w-14 mb-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={1.5}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
                />
              </svg>
              <span className="text-lg font-semibold tracking-wide">
                PDF Similarity
              </span>
              <p className="text-sm text-center mt-2 leading-snug opacity-90">
                Extract images from PDF and compare with images from Can Tho
                University Science Journal and Publishing House.
              </p>
            </div>
          </div>
        </Link>
      </div>
    </div>
  );
}

export default HomePage;
